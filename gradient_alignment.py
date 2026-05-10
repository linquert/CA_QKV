"""Gradient-alignment diagnostics for routing updates.

This module asks whether the *actual optimizer update direction* induced by a
training batch is aligned with an idealized direction that would improve QK
routing. For QK routing, the ideal diagnostic objective is the true-offset QK
margin:

    mean score(query y_i, key x_j where j-i is a true offset)
  - mean score(query y_i, key x_j where j-i is a false offset)

The actual loss gradient points in the direction that *increases* loss, while an
optimizer step such as SGD moves approximately in the opposite direction. The
most interpretable cosine is therefore:

    cosine(-grad_loss, grad_qk_margin)

A positive value means the batch update is routing-aligned: taking the training
step should tend to increase the true-offset QK margin.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .metrics import _offsets_for_positions


def _as_long_tensor(values: Sequence[int], device: torch.device, max_len: int) -> torch.Tensor:
    valid = [int(v) for v in values if 0 <= int(v) < max_len]
    if not valid:
        raise ValueError("No valid token positions available for QK alignment metric")
    return torch.tensor(valid, dtype=torch.long, device=device)


def differentiable_qk_true_false_margin(
    q: torch.Tensor,
    k: torch.Tensor,
    query_token_positions: Sequence[int],
    key_token_positions: Sequence[int],
    true_offsets: Sequence[int],
    head_idx: int | None = None,
) -> torch.Tensor:
    """Differentiable true-minus-false QK score margin.

    q/k are expected to have shape [B, H, T, d_head]. The returned scalar remains
    connected to the model parameters through q and k, so it can be used with
    torch.autograd.grad.
    """
    device = q.device
    T = int(q.shape[2])
    qpos = _as_long_tensor(query_token_positions, device, T)
    kpos = _as_long_tensor(key_token_positions, device, T)
    qsub = q.index_select(2, qpos)  # [B,H,Lq,Dh]
    ksub = k.index_select(2, kpos)  # [B,H,Lk,Dh]
    scores = torch.matmul(qsub, ksub.transpose(-2, -1)) / (q.shape[-1] ** 0.5)  # [B,H,Lq,Lk]

    offsets = _offsets_for_positions(list(range(qpos.numel())), list(range(kpos.numel()))).to(device)
    true_mask = torch.zeros_like(offsets, dtype=torch.bool)
    for off in true_offsets:
        true_mask |= offsets == int(off)
    false_mask = ~true_mask
    if not true_mask.any() or not false_mask.any():
        raise ValueError("Need both true and false offset pairs to compute QK margin")

    if head_idx is not None:
        scores = scores[:, int(head_idx), :, :]  # [B,Lq,Lk]
    # Boolean indexing flattens Lq/Lk selected entries but keeps batch/head dims.
    true_score = scores[..., true_mask].mean()
    false_score = scores[..., false_mask].mean()
    return true_score - false_score


def _qk_params_for_layer(model, layer_idx: int) -> Dict[str, torch.nn.Parameter]:
    block = model.blocks[int(layer_idx)]
    return {
        f"layer_{layer_idx}/W_Q": block.attn.W_Q.weight,
        f"layer_{layer_idx}/W_K": block.attn.W_K.weight,
    }


def _clone_param_grads(params: Mapping[str, torch.nn.Parameter]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for name, p in params.items():
        if p.grad is None:
            out[name] = torch.zeros_like(p.detach())
        else:
            out[name] = p.grad.detach().clone()
    return out


def _flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    parts = [t.detach().reshape(-1) for t in tensors]
    if not parts:
        raise ValueError("Cannot flatten empty tensor list")
    return torch.cat(parts, dim=0)


def _safe_cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.detach().reshape(1, -1)
    b = b.detach().reshape(1, -1)
    if float(a.norm().item()) < eps or float(b.norm().item()) < eps:
        return 0.0
    return float(F.cosine_similarity(a, b, dim=-1).item())


def _head_row_slice(t: torch.Tensor, head_idx: int, d_head: int) -> torch.Tensor:
    start = int(head_idx) * int(d_head)
    end = start + int(d_head)
    # W_Q/W_K are Linear weights with shape [out_features, in_features]; head
    # identity lives in the output-feature row block.
    return t[start:end, :]


def _parse_heads(heads_cfg, n_heads: int) -> List[int]:
    if heads_cfg is None or heads_cfg == "all":
        return list(range(n_heads))
    if isinstance(heads_cfg, int):
        return [heads_cfg]
    return [int(h) for h in heads_cfg]


def _selected_actual_vectors(
    actual_grads: Mapping[str, torch.Tensor],
    ideal_grads: Mapping[str, torch.Tensor],
    param_names: Sequence[str],
    head_idx: int | None,
    d_head: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    actual_parts = []
    ideal_parts = []
    for name in param_names:
        ag = actual_grads[name]
        ig = ideal_grads[name]
        if head_idx is not None:
            ag = _head_row_slice(ag, head_idx, d_head)
            ig = _head_row_slice(ig, head_idx, d_head)
        # Actual optimizer step direction is approximately -grad_loss.
        actual_parts.append(-ag)
        ideal_parts.append(ig)
    return _flatten_tensors(actual_parts), _flatten_tensors(ideal_parts)


def compute_routing_gradient_alignment(
    *,
    model,
    batch: Mapping[str, torch.Tensor],
    tokenizer,
    row_length: int,
    true_offsets: Sequence[int],
    layer_idx: int = 0,
    heads="all",
    include_wq: bool = True,
    include_wk: bool = True,
    use_eval_mode: bool = False,
) -> Dict[str, float]:
    """Compare loss-gradient update direction to ideal QK-routing direction.

    Preconditions:
      - The caller has already run loss.backward() for the current training
        batch, so W_Q/W_K `.grad` buffers contain the actual loss gradients.
      - This function must not mutate those `.grad` buffers before the optimizer
        step. It uses `torch.autograd.grad` on a separate forward pass, which
        returns ideal gradients without writing to parameter `.grad`.
    """
    if not include_wq and not include_wk:
        return {}

    layer_idx = int(layer_idx)
    block = model.blocks[layer_idx]
    n_heads = int(block.attn.n_heads)
    d_head = int(block.attn.d_head)
    head_list = _parse_heads(heads, n_heads)

    params = _qk_params_for_layer(model, layer_idx)
    param_names = []
    if include_wq:
        param_names.append(f"layer_{layer_idx}/W_Q")
    if include_wk:
        param_names.append(f"layer_{layer_idx}/W_K")
    selected_params = [params[name] for name in param_names]
    actual_grads = _clone_param_grads({name: params[name] for name in param_names})

    was_training = model.training
    if use_eval_mode:
        model.eval()
    try:
        input_ids = batch["input_ids"]
        logits, cache = model(input_ids, return_cache=True)
        layer = cache[f"layer_{layer_idx}"]
        q = layer["q"]
        k = layer["k"]
        qpos = tokenizer.pred_positions_for_y(int(row_length))
        kpos = tokenizer.x_token_positions(int(row_length))

        metrics: Dict[str, float] = {}

        # All-head objective first.
        objective_all = differentiable_qk_true_false_margin(
            q, k, qpos, kpos, true_offsets=true_offsets, head_idx=None
        )
        ideal_all_tuple = torch.autograd.grad(
            objective_all,
            selected_params,
            retain_graph=bool(head_list),
            allow_unused=True,
        )
        ideal_all = {
            name: (g.detach().clone() if g is not None else torch.zeros_like(params[name].detach()))
            for name, g in zip(param_names, ideal_all_tuple)
        }
        actual_vec, ideal_vec = _selected_actual_vectors(actual_grads, ideal_all, param_names, None, d_head)
        metrics[f"grad_alignment/routing/layer_{layer_idx}/all_heads/update_cos"] = _safe_cosine(actual_vec, ideal_vec)
        metrics[f"grad_alignment/routing/layer_{layer_idx}/all_heads/loss_grad_cos"] = _safe_cosine(-actual_vec, ideal_vec)
        metrics[f"grad_alignment/routing/layer_{layer_idx}/all_heads/actual_update_norm"] = float(actual_vec.norm().item())
        metrics[f"grad_alignment/routing/layer_{layer_idx}/all_heads/ideal_norm"] = float(ideal_vec.norm().item())
        metrics[f"grad_alignment/routing/layer_{layer_idx}/all_heads/objective_margin"] = float(objective_all.detach().item())

        # Per-head objectives. Each ideal gradient is mostly supported on that
        # head's W_Q/W_K row block, but we slice both actual and ideal to make
        # the cosine specifically head-local.
        for idx, h in enumerate(head_list):
            objective_h = differentiable_qk_true_false_margin(
                q, k, qpos, kpos, true_offsets=true_offsets, head_idx=int(h)
            )
            retain = idx < len(head_list) - 1
            ideal_tuple = torch.autograd.grad(
                objective_h,
                selected_params,
                retain_graph=retain,
                allow_unused=True,
            )
            ideal = {
                name: (g.detach().clone() if g is not None else torch.zeros_like(params[name].detach()))
                for name, g in zip(param_names, ideal_tuple)
            }
            av, iv = _selected_actual_vectors(actual_grads, ideal, param_names, int(h), d_head)
            base = f"grad_alignment/routing/layer_{layer_idx}/head_{h}"
            metrics[f"{base}/update_cos"] = _safe_cosine(av, iv)
            metrics[f"{base}/loss_grad_cos"] = _safe_cosine(-av, iv)
            metrics[f"{base}/actual_update_norm"] = float(av.norm().item())
            metrics[f"{base}/ideal_norm"] = float(iv.norm().item())
            metrics[f"{base}/objective_margin"] = float(objective_h.detach().item())

        return metrics
    finally:
        if use_eval_mode and was_training:
            model.train()
