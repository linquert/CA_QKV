"""Linear probes over QKV/head/MLP activations."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def fit_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epochs: int = 50,
    lr: float = 1e-2,
    batch_size: int = 512,
    device: str | torch.device = "cpu",
) -> Tuple[LinearProbe, float]:
    """Train a quick linear probe and return heldout accuracy."""
    device = torch.device(device)
    features = features.float()
    labels = labels.long()
    n = features.shape[0]
    perm = torch.randperm(n)
    split = max(1, int(0.8 * n))
    train_idx, val_idx = perm[:split], perm[split:]
    if len(val_idx) == 0:
        val_idx = train_idx
    train_ds = TensorDataset(features[train_idx], labels[train_idx])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    probe = LinearProbe(features.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(probe(xb), yb)
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = probe(features[val_idx].to(device))
        acc = (logits.argmax(dim=-1).cpu() == labels[val_idx]).float().mean().item()
    return probe, float(acc)


@torch.no_grad()
def collect_probe_features(
    model,
    dataloader,
    tokenizer,
    row_length: int,
    device,
    max_batches: int = 8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collect per-cell features from first layer cache.

    Returns a dict keyed by feature kind with values containing features and labels.
    First implementation uses layer_0 for simplicity.
    """
    model.eval()
    xs_v = []
    xs_head = []
    xs_mlp = []
    state_labels = []
    pos_labels = []
    nb_labels = []
    rule_out_labels = []

    x_positions = tokenizer.x_token_positions(row_length)
    y_pred_positions = tokenizer.pred_positions_for_y(row_length)

    for bidx, batch in enumerate(dataloader):
        if bidx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        logits, cache = model(input_ids, return_cache=True)
        layer = cache["layer_0"]
        v = layer["v"]  # [B,H,T,Dh]
        head_out = layer["head_out"]  # [B,H,T,Dh]
        mlp_post = layer["mlp_post"]  # [B,T,Dmlp]
        B, H, T, Dh = v.shape

        # Source-cell V features: one feature per (example, head, x-position)
        for h in range(H):
            feats = v[:, h, x_positions, :].reshape(-1, Dh).detach().cpu()
            xs_v.append((h, feats))

        # Head output at y-prediction positions: one feature per (example, head, y-position)
        for h in range(H):
            feats = head_out[:, h, y_pred_positions, :].reshape(-1, Dh).detach().cpu()
            xs_head.append((h, feats))

        # MLP features at y-prediction positions.
        xs_mlp.append(mlp_post[:, y_pred_positions, :].reshape(-1, mlp_post.shape[-1]).detach().cpu())

        rows = batch["row"]
        next_rows = batch["next_row"]
        nbs = batch["neighborhood_ids"]
        B2, L = rows.shape
        state_labels.append(rows.reshape(-1).detach().cpu())
        pos_labels.append(torch.arange(L).repeat(B2).detach().cpu())
        nb_labels.append(nbs.reshape(-1).detach().cpu())
        rule_out_labels.append(next_rows.reshape(-1).detach().cpu())

    labels = {
        "state": torch.cat(state_labels),
        "position": torch.cat(pos_labels),
        "neighborhood": torch.cat(nb_labels),
        "rule_output": torch.cat(rule_out_labels),
    }

    # Re-group head-indexed features.
    v_by_head = {}
    head_by_head = {}
    for h, feats in xs_v:
        v_by_head.setdefault(h, []).append(feats)
    for h, feats in xs_head:
        head_by_head.setdefault(h, []).append(feats)

    return {
        "v": {f"head_{h}": torch.cat(parts) for h, parts in v_by_head.items()},
        "head_out": {f"head_{h}": torch.cat(parts) for h, parts in head_by_head.items()},
        "mlp_post": {"layer_0": torch.cat(xs_mlp)},
        "labels": labels,
    }


def run_standard_probes(
    model,
    dataloader,
    tokenizer,
    row_length: int,
    device,
    max_batches: int = 8,
    probe_epochs: int = 40,
) -> Dict[str, float]:
    data = collect_probe_features(model, dataloader, tokenizer, row_length, device, max_batches=max_batches)
    metrics = {}
    for head_name, feats in data["v"].items():
        _, acc_state = fit_probe(feats, data["labels"]["state"], 2, epochs=probe_epochs, device=device)
        _, acc_pos = fit_probe(feats, data["labels"]["position"], row_length, epochs=probe_epochs, device=device)
        metrics[f"probe/v_state_acc/layer_0/{head_name}"] = acc_state
        metrics[f"probe/v_position_acc/layer_0/{head_name}"] = acc_pos
    for head_name, feats in data["head_out"].items():
        _, acc_state = fit_probe(feats, data["labels"]["rule_output"], 2, epochs=probe_epochs, device=device)
        metrics[f"probe/head_out_rule_output_acc/layer_0/{head_name}"] = acc_state
    mlp_feats = data["mlp_post"]["layer_0"]
    _, acc_nb = fit_probe(mlp_feats, data["labels"]["neighborhood"], 8, epochs=probe_epochs, device=device)
    _, acc_rule = fit_probe(mlp_feats, data["labels"]["rule_output"], 2, epochs=probe_epochs, device=device)
    metrics["probe/mlp_neighborhood_acc/layer_0"] = acc_nb
    metrics["probe/mlp_rule_output_acc/layer_0"] = acc_rule
    return metrics
