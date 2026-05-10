"""Evaluation suite for controlled CA diagnostic sets."""
from __future__ import annotations

import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .ca_dataset import CADataset, CADatasetConfig, collate_ca_batch
from .ca_rules import CARule, true_offsets_for_rule
from .metrics import (
    masked_cross_entropy,
    masked_token_accuracy,
    row_and_cell_accuracy,
    row_and_cell_accuracy_masked,
    per_neighborhood_accuracy,
    per_neighborhood_accuracy_masked,
    per_position_accuracy,
    per_position_accuracy_masked,
    attention_entropy,
    attention_mass_by_cell_offset,
    true_false_attention_mass,
    attention_mass_to_token_groups,
    head_attention_divergence,
    head_offset_divergence,
    qk_scores_by_cell_offset,
    qk_true_false_margin,
    prefix_metrics,
)
from .probes import run_standard_probes
from .visualizations import make_epoch_visuals, save_prediction_text, plot_neighborhood_accuracy_bar
from .utils import ensure_dir, save_json


class CAEvalSuite:
    def __init__(self, config: dict, rule: CARule, tokenizer, device):
        self.config = config
        self.rule = rule
        self.tokenizer = tokenizer
        self.device = device
        self.row_length = int(config["data"]["row_length"])
        self.batch_size = int(config["data"].get("eval_batch_size", config["data"].get("batch_size", 128)))
        self.num_eval_examples = int(config["data"].get("num_eval_examples", 1024))
        self.boundary_condition = config["data"].get("boundary_condition", "zero")
        self.max_seq_len = int(config["model"].get("max_seq_len", 128))
        self.eval_loaders, self.eval_loader_meta = self._build_eval_loaders()
        # Fixed visualization/probe batch from canonical eval set.
        first_name = list(self.eval_loaders.keys())[0]
        self.fixed_batch = next(iter(self.eval_loaders[first_name]))
        self.dense_loader = self._build_dense_loader()

    def _build_eval_loaders(self):
        loaders = {}
        meta = {}
        lengths = self.config["data"].get("eval_lengths", [self.row_length])
        dists = self.config["data"].get("eval_distributions", ["balanced", "skewed_smooth", "random"])
        seed_base = int(self.config.get("seed", 0)) + 10_000
        for li, L in enumerate(lengths):
            for di, dist in enumerate(dists):
                cfg = CADatasetConfig(
                    row_length=int(L),
                    num_examples=self.num_eval_examples,
                    distribution=str(dist),
                    boundary_condition=self.boundary_condition,
                    max_seq_len=self.max_seq_len,
                )
                ds = CADataset(self.rule, self.tokenizer, cfg, split=f"eval_{L}_{dist}", seed=seed_base + li * 100 + di)
                name = f"len{L}_{dist}"
                loaders[name] = DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=collate_ca_batch,
                    num_workers=0,
                )
                meta[name] = {"row_length": int(L), "distribution": str(dist), "split": f"eval_{L}_{dist}"}
        return loaders, meta

    def _build_dense_loader(self):
        """Small fixed exact-balanced diagnostic loader for dense phase tracking.

        This uses single-cell supervision so each example contributes one known
        local neighborhood to the diagnostic. It is intentionally small enough to
        run frequently during early training.
        """
        dense_cfg = self.config.get("dense_eval", {})
        examples_per_nb = int(dense_cfg.get("examples_per_neighborhood", 16))
        num_examples = int(dense_cfg.get("num_examples", 8 * examples_per_nb))
        distribution = str(dense_cfg.get("distribution", "single_cell_balanced"))
        cfg = CADatasetConfig(
            row_length=self.row_length,
            num_examples=num_examples,
            distribution=distribution,
            boundary_condition=self.boundary_condition,
            max_seq_len=self.max_seq_len,
        )
        ds = CADataset(
            self.rule,
            self.tokenizer,
            cfg,
            split="dense_eval",
            seed=int(self.config.get("seed", 0)) + 50_000,
        )
        return DataLoader(
            ds,
            batch_size=int(dense_cfg.get("batch_size", self.batch_size)),
            shuffle=False,
            collate_fn=collate_ca_batch,
            num_workers=0,
        )

    @torch.no_grad()
    def evaluate_behavior(self, model) -> Dict[str, float]:
        model.eval()
        all_metrics = {}
        for name, loader in self.eval_loaders.items():
            loss_sum = 0.0
            mask_sum = 0.0
            token_acc_sum = 0.0
            token_acc_count = 0
            pred_rows_all = []
            true_rows_all = []
            nb_ids_all = []
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                target_ids = batch["target_ids"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                logits, _ = model(input_ids, return_cache=False)
                loss = masked_cross_entropy(logits, target_ids, loss_mask)
                loss_sum += loss.item() * float(loss_mask.sum().item())
                mask_sum += float(loss_mask.sum().item())
                token_acc_sum += masked_token_accuracy(logits, target_ids, loss_mask)
                token_acc_count += 1

                L = int(self.eval_loader_meta[name]["row_length"])
                pred_rows = self.tokenizer.extract_predicted_y_bits_from_logits(logits, L).cpu()
                pred_rows_all.append(pred_rows)
                true_rows_all.append(batch["next_row"].cpu())
                nb_ids_all.append(batch["neighborhood_ids"].cpu())

            pred_rows_all = torch.cat(pred_rows_all, dim=0)
            true_rows_all = torch.cat(true_rows_all, dim=0)
            nb_ids_all = torch.cat(nb_ids_all, dim=0)
            row_acc, cell_acc = row_and_cell_accuracy(pred_rows_all, true_rows_all)
            metrics = {
                "loss": loss_sum / max(mask_sum, 1.0),
                "token_acc": token_acc_sum / max(token_acc_count, 1),
                "row_acc": row_acc,
                "cell_acc": cell_acc,
            }
            metrics.update(per_neighborhood_accuracy(pred_rows_all, true_rows_all, nb_ids_all))
            metrics.update(per_position_accuracy(pred_rows_all, true_rows_all))
            all_metrics.update(prefix_metrics(metrics, f"eval/{name}"))
        return all_metrics

    @torch.no_grad()
    def evaluate_attention_qk(self, model, max_batches: int = 2) -> Dict[str, float]:
        model.eval()
        # Use first eval loader with original train row length if available, otherwise first loader.
        loader_name = f"len{self.row_length}_balanced"
        if loader_name not in self.eval_loaders:
            loader_name = list(self.eval_loaders.keys())[0]
        loader = self.eval_loaders[loader_name]
        true_offsets = true_offsets_for_rule(self.rule)
        accum: Dict[str, List[float]] = {}
        for bi, batch in enumerate(loader):
            if bi >= max_batches:
                break
            input_ids = batch["input_ids"].to(self.device)
            logits, cache = model(input_ids, return_cache=True)
            layer = cache["layer_0"]
            attn = layer["attn"]
            q = layer["q"]
            k = layer["k"]
            qpos = self.tokenizer.pred_positions_for_y(self.row_length)
            kpos = self.tokenizer.x_token_positions(self.row_length)
            d = {}
            d.update(attention_mass_by_cell_offset(attn, qpos, kpos, max_abs_offset=8))
            d.update(true_false_attention_mass(attn, qpos, kpos, true_offsets=true_offsets))
            d.update(attention_mass_to_token_groups(attn, qpos, self.tokenizer.attention_token_groups(self.row_length)))
            d.update(head_attention_divergence(attn, qpos))
            d.update(head_offset_divergence(attn, qpos, kpos, max_abs_offset=8))
            d.update(qk_scores_by_cell_offset(q, k, qpos, kpos, max_abs_offset=8))
            d.update(qk_true_false_margin(q, k, qpos, kpos, true_offsets=true_offsets))
            ent = attention_entropy(attn)
            for h, val in enumerate(ent.tolist()):
                d[f"attn/head_{h}/entropy"] = float(val)
            for key, val in d.items():
                accum.setdefault(key, []).append(float(val))
        metrics = {f"diagnostic/{k}": float(sum(vals) / len(vals)) for k, vals in accum.items() if vals}
        return metrics

    @torch.no_grad()
    def evaluate_dense_neighborhoods(self, model) -> Dict[str, float]:
        """Dense, masked neighborhood/position diagnostic on exact-balanced data."""
        model.eval()
        loss_sum = 0.0
        mask_sum = 0.0
        token_acc_sum = 0.0
        token_acc_count = 0
        pred_rows_all = []
        true_rows_all = []
        nb_ids_all = []
        query_mask_all = []
        for batch in self.dense_loader:
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)
            logits, _ = model(input_ids, return_cache=False)
            loss = masked_cross_entropy(logits, target_ids, loss_mask)
            loss_sum += loss.item() * float(loss_mask.sum().item())
            mask_sum += float(loss_mask.sum().item())
            token_acc_sum += masked_token_accuracy(logits, target_ids, loss_mask)
            token_acc_count += 1
            pred_rows_all.append(self.tokenizer.extract_predicted_y_bits_from_logits(logits, self.row_length).cpu())
            true_rows_all.append(batch["next_row"].cpu())
            nb_ids_all.append(batch["neighborhood_ids"].cpu())
            query_mask_all.append(batch["query_mask"].cpu())

        pred_rows_all = torch.cat(pred_rows_all, dim=0)
        true_rows_all = torch.cat(true_rows_all, dim=0)
        nb_ids_all = torch.cat(nb_ids_all, dim=0)
        query_mask_all = torch.cat(query_mask_all, dim=0)
        row_acc, cell_acc = row_and_cell_accuracy_masked(pred_rows_all, true_rows_all, query_mask_all)
        metrics = {
            "loss": loss_sum / max(mask_sum, 1.0),
            "token_acc": token_acc_sum / max(token_acc_count, 1),
            "row_acc": row_acc,
            "cell_acc": cell_acc,
        }
        metrics.update(per_neighborhood_accuracy_masked(pred_rows_all, true_rows_all, nb_ids_all, query_mask_all))
        metrics.update(per_position_accuracy_masked(pred_rows_all, true_rows_all, query_mask_all))
        return prefix_metrics(metrics, "dense")

    @torch.no_grad()
    def evaluate_update_fingerprint(self, model) -> Dict[str, float]:
        """Small controlled diagnostic used before/after selected updates.

        This intentionally reuses the dense exact-balanced loader, plus optional
        attention/QK diagnostics, to produce a compact metric vector whose
        after-before difference is the semantic fingerprint of one optimizer step.
        """
        fp_cfg = self.config.get("delta", {})
        metrics: Dict[str, float] = {}

        dense = self.evaluate_dense_neighborhoods(model)
        for key, val in dense.items():
            # dense/neighborhood/101_acc -> fingerprint/neighborhood/101_acc
            stripped = key[len("dense/"):] if key.startswith("dense/") else key
            metrics[f"fingerprint/{stripped}"] = val

        if bool(fp_cfg.get("include_attention_qk", True)):
            max_batches = int(fp_cfg.get("max_attention_batches", 1))
            diag = self.evaluate_attention_qk(model, max_batches=max_batches)
            for key, val in diag.items():
                stripped = key[len("diagnostic/"):] if key.startswith("diagnostic/") else key
                metrics[f"fingerprint/{stripped}"] = val
        return metrics

    def evaluate_probes(self, model) -> Dict[str, float]:
        loader_name = f"len{self.row_length}_balanced"
        if loader_name not in self.eval_loaders:
            loader_name = list(self.eval_loaders.keys())[0]
        return run_standard_probes(
            model=model,
            dataloader=self.eval_loaders[loader_name],
            tokenizer=self.tokenizer,
            row_length=self.row_length,
            device=self.device,
            max_batches=int(self.config.get("probes", {}).get("max_batches", 8)),
            probe_epochs=int(self.config.get("probes", {}).get("epochs", 30)),
        )

    @torch.no_grad()
    def save_samples_and_figures(self, model, epoch: int, step: int, output_dir: str) -> List[str]:
        fig_dir = os.path.join(output_dir, "figures", f"epoch_{epoch:04d}")
        sample_dir = os.path.join(output_dir, "samples")
        ensure_dir(fig_dir)
        ensure_dir(sample_dir)
        batch = self.fixed_batch
        # Make visuals.
        paths = make_epoch_visuals(
            model=model,
            batch={k: v.to(self.device) if torch.is_tensor(v) and k in {"input_ids", "target_ids", "loss_mask"} else v for k, v in batch.items()},
            tokenizer=self.tokenizer,
            row_length=self.row_length,
            out_dir=fig_dir,
            epoch=epoch,
            device=self.device,
        )
        # Save text samples.
        model.eval()
        input_ids = batch["input_ids"].to(self.device)
        logits, _ = model(input_ids, return_cache=False)
        pred_rows = self.tokenizer.extract_predicted_y_bits_from_logits(logits, self.row_length).cpu()
        examples = []
        n = min(8, pred_rows.shape[0])
        for i in range(n):
            examples.append({
                "input_row": batch["row"][i].tolist(),
                "target_row": batch["next_row"][i].tolist(),
                "pred_row": pred_rows[i].tolist(),
                "neighborhood_ids": batch["neighborhood_ids"][i].tolist(),
            })
        txt_path = os.path.join(sample_dir, f"epoch_{epoch:04d}_step_{step:08d}_predictions.txt")
        save_prediction_text(examples, txt_path, header=f"Epoch {epoch}, step {step}, rule {self.rule.name}")
        paths.append(txt_path)
        # Neighborhood accuracy bar from first eval set for visual, using behavior metrics quick subset.
        behavior = self.evaluate_behavior(model)
        prefix = f"eval/len{self.row_length}_balanced/"
        nb_metrics = {k.replace(prefix, ""): v for k, v in behavior.items() if k.startswith(prefix + "neighborhood/")}
        if nb_metrics:
            nb_path = plot_neighborhood_accuracy_bar(nb_metrics, os.path.join(fig_dir, f"epoch_{epoch:04d}_neighborhood_acc.png"))
            paths.append(nb_path)
        return paths
