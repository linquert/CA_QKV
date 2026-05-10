"""Experiment logging wrapper for local files and W&B."""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

import torch

from .utils import ensure_dir, save_json


class ExperimentLogger:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config.get("output_dir", "outputs")
        ensure_dir(self.output_dir)
        self.use_wandb = bool(config.get("logging", {}).get("use_wandb", False))
        self.wandb = None
        if self.use_wandb:
            import wandb
            self.wandb = wandb
            wandb.init(
                project=config.get("logging", {}).get("project", "ca-qkv"),
                name=config.get("experiment_name", None),
                config=config,
            )

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        clean = {k: v for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
        if self.use_wandb:
            self.wandb.log(clean, step=step)

    def log_figures(self, paths: List[str], step: int) -> None:
        if not self.use_wandb:
            return
        imgs = {}
        for p in paths:
            key = "figures/" + os.path.splitext(os.path.basename(p))[0]
            imgs[key] = self.wandb.Image(p)
        if imgs:
            self.wandb.log(imgs, step=step)

    def save_checkpoint(self, model, optimizer, epoch: int, step: int, metrics: Dict[str, float] | None = None) -> str:
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        ensure_dir(ckpt_dir)
        path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}_step_{step:08d}.pt")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "step": step,
            "config": self.config,
            "metrics": metrics or {},
        }, path)
        if self.use_wandb and self.config.get("logging", {}).get("save_artifacts", False):
            artifact = self.wandb.Artifact(
                name=f"{self.config.get('experiment_name','ca_qkv')}_epoch_{epoch:04d}",
                type="checkpoint",
            )
            artifact.add_file(path)
            self.wandb.log_artifact(artifact)
        return path

    def finish(self):
        if self.use_wandb:
            self.wandb.finish()
