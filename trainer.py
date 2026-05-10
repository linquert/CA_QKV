"""Training loop coordinating updates, evals, logging, and saving."""
from __future__ import annotations

import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .ca_dataset import CADataset, CADatasetConfig, collate_ca_batch
from .ca_rules import build_rule, true_offsets_for_rule
from .delta import append_delta_jsonl, diff_metrics, summarize_batch_for_delta
from .evaluator import CAEvalSuite
from .hooks import snapshot_selected_weights, compute_gradient_metrics, compute_update_metrics
from .gradient_alignment import compute_routing_gradient_alignment
from .logger import ExperimentLogger
from .metrics import masked_cross_entropy, masked_token_accuracy
from .model import build_model_from_config
from .tokenizer import CATokenizer
from .utils import ensure_dir, get_device, seed_everything


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        seed_everything(int(config.get("seed", 0)))
        self.device = get_device(bool(config.get("prefer_cuda", True)))
        self.output_dir = config.get("output_dir", "outputs")
        ensure_dir(self.output_dir)

        max_row_length = max([int(config["data"]["row_length"])] + [int(x) for x in config["data"].get("eval_lengths", [])])
        self.tokenizer = CATokenizer(
            max_row_length=max_row_length,
            include_position_tokens=bool(config.get("tokenizer", {}).get("include_position_tokens", False)),
        )
        self.rule = build_rule(config["rule"])
        self.model = build_model_from_config(config, vocab_size=self.tokenizer.vocab_size).to(self.device)
        self.optimizer = self._build_optimizer()
        self.logger = ExperimentLogger(config)

        train_cfg = CADatasetConfig(
            row_length=int(config["data"]["row_length"]),
            num_examples=int(config["data"].get("num_train_examples", 20000)),
            distribution=str(config["data"].get("train_distribution", "balanced")),
            boundary_condition=str(config["data"].get("boundary_condition", "zero")),
            max_seq_len=int(config["model"].get("max_seq_len", 128)),
            smooth_flip_prob=float(config["data"].get("smooth_flip_prob", 0.08)),
        )
        self.train_dataset = CADataset(
            rule=self.rule,
            tokenizer=self.tokenizer,
            cfg=train_cfg,
            split="train",
            seed=int(config.get("seed", 0)),
        )
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=int(config["data"].get("batch_size", 128)),
            shuffle=True,
            collate_fn=collate_ca_batch,
            num_workers=int(config["data"].get("num_workers", 0)),
            pin_memory=torch.cuda.is_available(),
        )
        self.eval_suite = CAEvalSuite(config, self.rule, self.tokenizer, self.device)
        self.global_step = 0

    def _build_optimizer(self):
        train_cfg = self.config.get("training", {})
        lr = float(train_cfg.get("lr", 5e-4))
        wd = float(train_cfg.get("weight_decay", 0.01))
        opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
        if opt_name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=wd)
        if opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def _should_log_dense_eval(self) -> bool:
        cfg = self.config.get("dense_eval", {})
        if not bool(cfg.get("enabled", False)):
            return False
        step = self.global_step
        until = cfg.get("until_step", None)
        every = int(cfg.get("every_steps", 1))
        if until is not None and step <= int(until):
            return step % max(every, 1) == 0
        then_every = int(cfg.get("then_every_steps", cfg.get("after_until_every_steps", 0) or 0))
        return then_every > 0 and step % then_every == 0

    def _should_log_delta(self, step: int) -> bool:
        cfg = self.config.get("delta", {})
        if not bool(cfg.get("enabled", False)):
            return False
        until = cfg.get("until_step", None)
        every = int(cfg.get("every_steps", 1))
        if until is not None and step <= int(until):
            return step % max(every, 1) == 0
        then_every = int(cfg.get("then_every_steps", cfg.get("after_until_every_steps", 0) or 0))
        return then_every > 0 and step % then_every == 0

    def _should_compute_gradient_alignment(self, step: int) -> bool:
        cfg = self.config.get("gradient_alignment", {})
        if not bool(cfg.get("enabled", False)):
            return False
        until = cfg.get("until_step", None)
        every = int(cfg.get("every_steps", 50))
        if until is not None and step <= int(until):
            return step % max(every, 1) == 0
        then_every = int(cfg.get("then_every_steps", cfg.get("after_until_every_steps", every) or 0))
        return then_every > 0 and step % then_every == 0

    def _delta_jsonl_path(self) -> str:
        return os.path.join(self.output_dir, "deltas", "update_deltas.jsonl")

    def _run_delta_before_after(self, batch: Dict[str, torch.Tensor], step_after_update: int):
        """Evaluate fingerprint before and after exactly one optimizer step."""
        before_fp = self.eval_suite.evaluate_update_fingerprint(self.model)
        metrics = self.train_step(batch)
        after_fp = self.eval_suite.evaluate_update_fingerprint(self.model)
        delta_metrics = diff_metrics(before_fp, after_fp, prefix="delta")
        if bool(self.config.get("delta", {}).get("log_before_after", False)):
            self.logger.log_metrics({f"before/{k}": v for k, v in before_fp.items()}, step=step_after_update)
            self.logger.log_metrics({f"after/{k}": v for k, v in after_fp.items()}, step=step_after_update)
        self.logger.log_metrics(delta_metrics, step=step_after_update)
        if bool(self.config.get("delta", {}).get("save_jsonl", True)):
            record = {
                "step": int(step_after_update),
                "batch_features": summarize_batch_for_delta(batch),
                "before": before_fp,
                "after": after_fp,
                "delta": delta_metrics,
            }
            append_delta_jsonl(self._delta_jsonl_path(), record)
        metrics.update(delta_metrics)
        return metrics

    def train(self):
        epochs = int(self.config["training"].get("epochs", 50))
        eval_every_steps = int(self.config["training"].get("eval_every_steps", 500))
        log_every_steps = int(self.config["training"].get("log_every_steps", 50))
        save_every_epochs = int(self.config["training"].get("save_every_epochs", 5))
        full_eval_every_epochs = int(self.config["training"].get("full_eval_every_epochs", 1))

        try:
            for epoch in range(1, epochs + 1):
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{epochs}")
                for batch in pbar:
                    next_step = self.global_step + 1
                    if self._should_log_delta(next_step):
                        metrics = self._run_delta_before_after(batch, step_after_update=next_step)
                    else:
                        metrics = self.train_step(batch)
                    self.global_step = next_step
                    if self.global_step % log_every_steps == 0:
                        self.logger.log_metrics(metrics, step=self.global_step)
                        pbar.set_postfix({"loss": f"{metrics['train/loss']:.4f}"})
                    if self._should_log_dense_eval():
                        dense_metrics = self.eval_suite.evaluate_dense_neighborhoods(self.model)
                        self.logger.log_metrics(dense_metrics, step=self.global_step)
                    if self.global_step % eval_every_steps == 0:
                        diag = self.eval_suite.evaluate_attention_qk(self.model)
                        self.logger.log_metrics(diag, step=self.global_step)

                if full_eval_every_epochs > 0 and epoch % full_eval_every_epochs == 0:
                    self.end_of_epoch(epoch)
                if save_every_epochs > 0 and epoch % save_every_epochs == 0:
                    self.logger.save_checkpoint(self.model, self.optimizer, epoch, self.global_step)
        finally:
            self.logger.finish()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.model.train()
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        before = snapshot_selected_weights(self.model)
        logits, _ = self.model(batch["input_ids"], return_cache=False)
        loss = masked_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        tok_acc = masked_token_accuracy(logits.detach(), batch["target_ids"], batch["loss_mask"])
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_metrics = compute_gradient_metrics(self.model)
        if self._should_compute_gradient_alignment(self.global_step + 1):
            ga_cfg = self.config.get("gradient_alignment", {})
            alignment_metrics = compute_routing_gradient_alignment(
                model=self.model,
                batch=batch,
                tokenizer=self.tokenizer,
                row_length=int(self.config["data"]["row_length"]),
                true_offsets=true_offsets_for_rule(self.rule),
                layer_idx=int(ga_cfg.get("layer_idx", 0)),
                heads=ga_cfg.get("heads", "all"),
                include_wq=bool(ga_cfg.get("include_wq", True)),
                include_wk=bool(ga_cfg.get("include_wk", True)),
                use_eval_mode=bool(ga_cfg.get("use_eval_mode", False)),
            )
            grad_metrics.update(alignment_metrics)
        self.optimizer.step()
        update_metrics = compute_update_metrics(self.model, before)
        metrics = {
            "train/loss": float(loss.item()),
            "train/token_acc": float(tok_acc),
            "train/lr": float(self.optimizer.param_groups[0]["lr"]),
        }
        metrics.update(grad_metrics)
        metrics.update(update_metrics)
        return metrics

    def end_of_epoch(self, epoch: int):
        metrics = {}
        metrics.update(self.eval_suite.evaluate_behavior(self.model))
        metrics.update(self.eval_suite.evaluate_attention_qk(self.model))
        if bool(self.config.get("dense_eval", {}).get("log_at_epoch_end", True)):
            metrics.update(self.eval_suite.evaluate_dense_neighborhoods(self.model))
        if epoch % int(self.config.get("probes", {}).get("every_epochs", 1)) == 0:
            metrics.update(self.eval_suite.evaluate_probes(self.model))
        self.logger.log_metrics(metrics, step=self.global_step)
        if self.config.get("logging", {}).get("log_figures", True):
            paths = self.eval_suite.save_samples_and_figures(self.model, epoch, self.global_step, self.output_dir)
            pngs = [p for p in paths if p.lower().endswith(".png")]
            self.logger.log_figures(pngs, step=self.global_step)
