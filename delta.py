"""Update-level delta/fingerprint utilities.

The central object for update semantics is not just a metric at a checkpoint,
but the change in a controlled diagnostic metric caused by a single optimizer
step. This module provides small helpers for computing, logging, and storing
those per-update deltas.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Mapping

import torch

from .utils import ensure_dir


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x


def diff_metrics(
    before: Mapping[str, float],
    after: Mapping[str, float],
    prefix: str = "delta",
) -> Dict[str, float]:
    """Return after-before for numeric keys present in both dictionaries.

    NaNs are skipped. Key names are preserved under `prefix`, e.g.
    `fingerprint/neighborhood/101_acc` becomes
    `delta/fingerprint/neighborhood/101_acc`.
    """
    out: Dict[str, float] = {}
    for key, b in before.items():
        if key not in after:
            continue
        a = after[key]
        if not (_is_number(a) and _is_number(b)):
            continue
        out[f"{prefix}/{key}"] = float(a) - float(b)
    return out


def summarize_batch_for_delta(batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Create lightweight batch metadata for a JSONL delta record.

    The goal is to make each delta interpretable after the fact: what
    neighborhoods/positions were supervised by the batch that caused this update?
    """
    out: Dict[str, Any] = {}
    if "neighborhood_ids" in batch:
        nb = batch["neighborhood_ids"]
        if torch.is_tensor(nb):
            nb_cpu = nb.detach().cpu()
            if "query_mask" in batch and torch.is_tensor(batch["query_mask"]):
                mask = batch["query_mask"].detach().cpu() > 0
                nb_used = nb_cpu[mask]
            else:
                nb_used = nb_cpu.reshape(-1)
            counts = torch.bincount(nb_used.long(), minlength=8).tolist() if nb_used.numel() else [0] * 8
            out["supervised_neighborhood_counts"] = {f"{i:03b}": int(c) for i, c in enumerate(counts[:8])}
    if "query_mask" in batch and torch.is_tensor(batch["query_mask"]):
        qm = batch["query_mask"].detach().cpu()
        out["num_supervised_cells"] = int((qm > 0).sum().item())
        if qm.ndim == 2:
            pos_counts = (qm > 0).sum(dim=0).long().tolist()
            out["supervised_position_counts"] = {str(i): int(c) for i, c in enumerate(pos_counts)}
    if "row" in batch and torch.is_tensor(batch["row"]):
        out["batch_size"] = int(batch["row"].shape[0])
        out["row_length"] = int(batch["row"].shape[1]) if batch["row"].ndim >= 2 else None
    return out


def append_delta_jsonl(
    path: str,
    record: Mapping[str, Any],
) -> None:
    """Append one update-delta record to a JSONL file."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
