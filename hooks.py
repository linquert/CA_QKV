"""Gradient and update snapshot utilities."""
from __future__ import annotations

from typing import Dict

import torch


SELECTED_SUFFIXES = [
    "attn.W_Q.weight",
    "attn.W_K.weight",
    "attn.W_V.weight",
    "attn.W_O.weight",
    "mlp.0.weight",
    "mlp.2.weight",
]


def selected_named_parameters(model):
    for name, p in model.named_parameters():
        if any(name.endswith(suf) for suf in SELECTED_SUFFIXES):
            yield name, p


def snapshot_selected_weights(model) -> Dict[str, torch.Tensor]:
    return {name: p.detach().clone() for name, p in selected_named_parameters(model)}


def compute_update_metrics(model, before: Dict[str, torch.Tensor]) -> Dict[str, float]:
    out = {}
    for name, p in selected_named_parameters(model):
        if name in before:
            delta = p.detach() - before[name].to(p.device)
            key = "update_norm/" + name.replace("blocks.", "layer_").replace(".", "/")
            out[key] = float(delta.norm().item())
    return out


def compute_gradient_metrics(model) -> Dict[str, float]:
    out = {}
    total_sq = 0.0
    for name, p in selected_named_parameters(model):
        if p.grad is None:
            val = 0.0
        else:
            val = float(p.grad.detach().norm().item())
            total_sq += val * val
        key = "grad_norm/" + name.replace("blocks.", "layer_").replace(".", "/")
        out[key] = val
    out["grad_norm/selected_total"] = total_sq ** 0.5
    return out
