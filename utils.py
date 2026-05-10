"""Shared utilities."""
from __future__ import annotations

import json
import os
import random
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    out = SimpleNamespace()
    for k, v in d.items():
        setattr(out, k, dict_to_namespace(v) if isinstance(v, dict) else v)
    return out


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    items = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, key))
        else:
            items[key] = v
    return items


def get_device(prefer_cuda: bool = True) -> torch.device:
    return torch.device("cuda" if prefer_cuda and torch.cuda.is_available() else "cpu")


def detach_to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    if isinstance(x, dict):
        return {k: detach_to_cpu(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(detach_to_cpu(v) for v in x)
    return x
