"""Visualization helpers for epoch-level observability."""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

from .utils import ensure_dir


def _savefig(path: str):
    ensure_dir(os.path.dirname(path))
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_attention_matrix(attn: torch.Tensor, path: str, title: str = "Attention") -> str:
    """attn [Q,K] numpy/torch."""
    arr = attn.detach().cpu().float().numpy() if torch.is_tensor(attn) else np.asarray(attn)
    plt.figure(figsize=(7, 6))
    plt.imshow(arr, aspect="auto", origin="upper")
    plt.colorbar(label="attention")
    plt.xlabel("key token position")
    plt.ylabel("query token position")
    plt.title(title)
    return _savefig(path)


def plot_offset_bars(offset_values: Dict[int, float], path: str, title: str = "Attention mass by relative offset") -> str:
    offsets = sorted(offset_values.keys())
    vals = [offset_values[o] for o in offsets]
    plt.figure(figsize=(8, 4))
    plt.bar([str(o) for o in offsets], vals)
    plt.xlabel("cell offset = key index - query index")
    plt.ylabel("mean mass / score")
    plt.title(title)
    return _savefig(path)


def plot_qk_matrix(qk_scores: torch.Tensor, path: str, title: str = "QK score matrix") -> str:
    arr = qk_scores.detach().cpu().float().numpy() if torch.is_tensor(qk_scores) else np.asarray(qk_scores)
    plt.figure(figsize=(6, 5))
    plt.imshow(arr, aspect="auto", origin="upper")
    plt.colorbar(label="QK score")
    plt.xlabel("key input cell")
    plt.ylabel("query output cell")
    plt.title(title)
    return _savefig(path)


def plot_prediction_grid(input_row, target_row, pred_row, path: str, title: str = "Prediction grid") -> str:
    x = np.asarray(input_row, dtype=int)
    y = np.asarray(target_row, dtype=int)
    p = np.asarray(pred_row, dtype=int)
    err = (p != y).astype(int)
    arr = np.stack([x, y, p, err], axis=0)
    plt.figure(figsize=(max(8, len(x) * 0.35), 2.4))
    plt.imshow(arr, aspect="auto", interpolation="nearest")
    plt.yticks([0, 1, 2, 3], ["input", "target", "pred", "error"])
    plt.xticks(range(len(x)), range(len(x)), fontsize=7)
    plt.title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            plt.text(j, i, str(arr[i, j]), ha="center", va="center", fontsize=7)
    return _savefig(path)


def plot_metric_history(history: Dict[str, List[float]], path: str, title: str, ylabel: str = "value") -> str:
    plt.figure(figsize=(8, 4))
    for name, vals in history.items():
        plt.plot(vals, label=name)
    plt.xlabel("logging point")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    return _savefig(path)


def plot_neighborhood_accuracy_bar(metrics: Dict[str, float], path: str, title: str = "Neighborhood accuracy") -> str:
    labels = [f"{i:03b}" for i in range(8)]
    vals = [metrics.get(f"neighborhood/{lab}_acc", np.nan) for lab in labels]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, vals)
    plt.ylim(0, 1.0)
    plt.xlabel("neighborhood")
    plt.ylabel("accuracy")
    plt.title(title)
    return _savefig(path)


def save_prediction_text(examples: List[Dict], path: str, header: str = "") -> str:
    ensure_dir(os.path.dirname(path))
    lines = []
    if header:
        lines.append(header)
        lines.append("")
    for idx, ex in enumerate(examples):
        x = ex["input_row"]
        y = ex["target_row"]
        p = ex["pred_row"]
        err = ["X" if int(a) != int(b) else "." for a, b in zip(p, y)]
        lines.append(f"Example {idx}")
        lines.append("x:      " + " ".join(map(str, x)))
        lines.append("target: " + " ".join(map(str, y)))
        lines.append("pred:   " + " ".join(map(str, p)))
        lines.append("error:  " + " ".join(err))
        if "neighborhood_ids" in ex:
            lines.append("nb:     " + " ".join(f"{int(n):03b}" for n in ex["neighborhood_ids"]))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def make_epoch_visuals(
    model,
    batch,
    tokenizer,
    row_length: int,
    out_dir: str,
    epoch: int,
    device,
    max_abs_offset: int = 8,
) -> List[str]:
    """Generate a core set of figures from one fixed batch."""
    ensure_dir(out_dir)
    model.eval()
    paths = []
    with torch.no_grad():
        input_ids = batch["input_ids"].to(device)
        logits, cache = model(input_ids, return_cache=True)
        pred_rows = tokenizer.extract_predicted_y_bits_from_logits(logits, row_length).cpu()
        layer = cache["layer_0"]
        attn = layer["attn"][0]  # [H,T,T]
        q = layer["q"][0]
        k = layer["k"][0]

        # Prediction grid for first example.
        p0 = plot_prediction_grid(
            batch["row"][0].tolist(),
            batch["next_row"][0].tolist(),
            pred_rows[0].tolist(),
            os.path.join(out_dir, f"epoch_{epoch:04d}_prediction_grid.png"),
            title=f"Epoch {epoch} prediction",
        )
        paths.append(p0)

        # Full attention matrix per head.
        for h in range(attn.shape[0]):
            p = plot_attention_matrix(
                attn[h],
                os.path.join(out_dir, f"epoch_{epoch:04d}_attn_head_{h}.png"),
                title=f"Epoch {epoch} layer0 head{h} attention",
            )
            paths.append(p)

        # QK cell matrix per head.
        qpos = torch.tensor(tokenizer.pred_positions_for_y(row_length), device=device)
        kpos = torch.tensor(tokenizer.x_token_positions(row_length), device=device)
        qsub = layer["q"][0].index_select(1, qpos)  # [H,L,D]
        ksub = layer["k"][0].index_select(1, kpos)  # [H,L,D]
        scores = torch.matmul(qsub, ksub.transpose(-2, -1)) / (qsub.shape[-1] ** 0.5)
        for h in range(scores.shape[0]):
            p = plot_qk_matrix(
                scores[h],
                os.path.join(out_dir, f"epoch_{epoch:04d}_qk_head_{h}.png"),
                title=f"Epoch {epoch} layer0 head{h} QK scores over cells",
            )
            paths.append(p)
    return paths
