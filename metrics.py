"""Behavioral, attention/QK, and helper metrics."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F


def masked_cross_entropy(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Cross entropy over masked positions.

    logits: [B,T,V], target_ids: [B,T], loss_mask: [B,T]
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(logits.reshape(B * T, V), target_ids.reshape(B * T), reduction="none")
    loss = loss.view(B, T)
    denom = loss_mask.sum().clamp_min(1.0)
    return (loss * loss_mask).sum() / denom


def masked_token_accuracy(logits: torch.Tensor, target_ids: torch.Tensor, loss_mask: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    correct = ((preds == target_ids) * (loss_mask > 0)).sum().item()
    total = (loss_mask > 0).sum().item()
    return float(correct / max(total, 1))


def row_and_cell_accuracy(pred_rows: torch.Tensor, true_rows: torch.Tensor) -> Tuple[float, float]:
    """pred_rows/true_rows [B,L] with values 0/1."""
    cell = (pred_rows == true_rows).float().mean().item()
    row = (pred_rows == true_rows).all(dim=1).float().mean().item()
    return float(row), float(cell)


def row_and_cell_accuracy_masked(
    pred_rows: torch.Tensor,
    true_rows: torch.Tensor,
    query_mask: torch.Tensor,
) -> Tuple[float, float]:
    """Masked row/cell accuracy.

    For full-row datasets, query_mask is all ones and this equals ordinary cell
    accuracy. For single-cell diagnostic datasets, only the queried/supervised
    cell contributes. Row accuracy is interpreted as: all queried cells in the
    example are correct.
    """
    mask = query_mask > 0
    correct = pred_rows == true_rows
    cell = correct[mask].float().mean().item() if mask.any() else float("nan")
    # For each row, check all queried cells; ignore rows with no queried cells.
    per_row_has_query = mask.any(dim=1)
    per_row_correct = torch.ones(pred_rows.shape[0], dtype=torch.bool, device=pred_rows.device)
    per_row_correct[per_row_has_query] = (correct | ~mask)[per_row_has_query].all(dim=1)
    row = per_row_correct[per_row_has_query].float().mean().item() if per_row_has_query.any() else float("nan")
    return float(row), float(cell)


def per_neighborhood_accuracy(pred_rows: torch.Tensor, true_rows: torch.Tensor, neighborhood_ids: torch.Tensor) -> Dict[str, float]:
    out = {}
    for idx in range(8):
        mask = neighborhood_ids == idx
        if mask.sum().item() == 0:
            out[f"neighborhood/{idx:03b}_acc"] = float("nan")
        else:
            out[f"neighborhood/{idx:03b}_acc"] = float((pred_rows[mask] == true_rows[mask]).float().mean().item())
    return out


def per_neighborhood_accuracy_masked(
    pred_rows: torch.Tensor,
    true_rows: torch.Tensor,
    neighborhood_ids: torch.Tensor,
    query_mask: torch.Tensor,
) -> Dict[str, float]:
    out = {}
    base_mask = query_mask > 0
    for idx in range(8):
        mask = (neighborhood_ids == idx) & base_mask
        if mask.sum().item() == 0:
            out[f"neighborhood/{idx:03b}_acc"] = float("nan")
        else:
            out[f"neighborhood/{idx:03b}_acc"] = float((pred_rows[mask] == true_rows[mask]).float().mean().item())
    return out


def per_position_accuracy(pred_rows: torch.Tensor, true_rows: torch.Tensor) -> Dict[str, float]:
    L = true_rows.shape[1]
    return {f"position/{i}_acc": float((pred_rows[:, i] == true_rows[:, i]).float().mean().item()) for i in range(L)}


def per_position_accuracy_masked(
    pred_rows: torch.Tensor,
    true_rows: torch.Tensor,
    query_mask: torch.Tensor,
) -> Dict[str, float]:
    L = true_rows.shape[1]
    out = {}
    correct = pred_rows == true_rows
    mask = query_mask > 0
    for i in range(L):
        m = mask[:, i]
        out[f"position/{i}_acc"] = float(correct[m, i].float().mean().item()) if m.any() else float("nan")
    return out


def attention_entropy(attn: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Entropy over key dimension. attn [B,H,T,T]. Returns [H] mean entropy."""
    a = attn.clamp_min(eps)
    ent = -(a * a.log()).sum(dim=-1)
    return ent.mean(dim=(0, 2))


def _offsets_for_positions(query_positions: List[int], key_positions: List[int]) -> torch.Tensor:
    # offset = key_cell_index - query_cell_index, not token index difference.
    q_idx = torch.arange(len(query_positions)).view(-1, 1)
    k_idx = torch.arange(len(key_positions)).view(1, -1)
    return k_idx - q_idx


def attention_mass_by_cell_offset(
    attn: torch.Tensor,
    query_token_positions: List[int],
    key_token_positions: List[int],
    max_abs_offset: int = 8,
) -> Dict[str, float]:
    """Mean per-query attention mass by cell offset for each head.

    Values are naturally in [0, 1] and comparable across row lengths.
    """
    device = attn.device
    T = attn.shape[-1]
    qpos = torch.tensor([p for p in query_token_positions if p < T], device=device, dtype=torch.long)
    kpos = torch.tensor([p for p in key_token_positions if p < T], device=device, dtype=torch.long)
    sub = attn.index_select(2, qpos).index_select(3, kpos)  # [B,H,Lq,Lk]
    offsets = _offsets_for_positions(list(range(len(qpos))), list(range(len(kpos)))).to(device)  # [Lq,Lk]
    out = {}
    for h in range(sub.shape[1]):
        for off in range(-max_abs_offset, max_abs_offset + 1):
            mask = offsets == off
            valid_queries = mask.any(dim=-1)
            if not valid_queries.any():
                val = 0.0
            else:
                mass_per_query = (sub[:, h, :, :] * mask.float()).sum(dim=-1)
                val = mass_per_query[:, valid_queries].mean().item()
            out[f"attn/head_{h}/offset_{off}_mass"] = float(val)
    return out


def true_false_attention_mass(
    attn: torch.Tensor,
    query_token_positions: List[int],
    key_token_positions: List[int],
    true_offsets: List[int],
) -> Dict[str, float]:
    """Mean per-query attention mass on true vs false x-cell dependency offsets."""
    device = attn.device
    T = attn.shape[-1]
    qpos = torch.tensor([p for p in query_token_positions if p < T], device=device, dtype=torch.long)
    kpos = torch.tensor([p for p in key_token_positions if p < T], device=device, dtype=torch.long)
    sub = attn.index_select(2, qpos).index_select(3, kpos)  # [B,H,Lq,Lk]
    offsets = _offsets_for_positions(list(range(len(qpos))), list(range(len(kpos)))).to(device)
    true_mask = torch.zeros_like(offsets, dtype=torch.bool)
    for off in true_offsets:
        true_mask |= offsets == off
    false_mask = ~true_mask
    out = {}
    true_mask_f = true_mask.float()
    false_mask_f = false_mask.float()
    for h in range(sub.shape[1]):
        true_val = (sub[:, h, :, :] * true_mask_f).sum(dim=-1).mean().item()
        false_val = (sub[:, h, :, :] * false_mask_f).sum(dim=-1).mean().item()
        out[f"attn/head_{h}/true_offset_mass"] = float(true_val)
        out[f"attn/head_{h}/false_offset_mass"] = float(false_val)
    return out


def attention_mass_to_token_groups(
    attn: torch.Tensor,
    query_token_positions: List[int],
    token_groups: Dict[str, List[int]],
) -> Dict[str, float]:
    """Mean per-query attention mass to named token-position groups.

    This captures attention sinks such as <BOS>, <SEP>, and <Y>. Groups can
    overlap; for example, `structural` may be the union of several marker tokens.
    """
    device = attn.device
    T = attn.shape[-1]
    qpos = torch.tensor([p for p in query_token_positions if p < T], device=device, dtype=torch.long)
    q_attn = attn.index_select(2, qpos)  # [B,H,Lq,T]
    out = {}
    for h in range(q_attn.shape[1]):
        for group_name, positions in token_groups.items():
            valid = sorted({int(p) for p in positions if 0 <= int(p) < T})
            if not valid:
                val = 0.0
            else:
                idx = torch.tensor(valid, device=device, dtype=torch.long)
                val = q_attn[:, h, :, :].index_select(-1, idx).sum(dim=-1).mean().item()
            out[f"attn/head_{h}/group/{group_name}_mass"] = float(val)
    return out


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p / m).log()).sum(dim=-1) + 0.5 * (q * (q / m).log()).sum(dim=-1)


def head_attention_divergence(
    attn: torch.Tensor,
    query_token_positions: List[int],
) -> Dict[str, float]:
    """Pairwise JS divergence between heads' full attention distributions.

    Averaged over batch and y-query positions. Low values indicate redundancy;
    high values indicate specialization/competition/collapse differences.
    """
    device = attn.device
    T = attn.shape[-1]
    qpos = torch.tensor([p for p in query_token_positions if p < T], device=device, dtype=torch.long)
    sub = attn.index_select(2, qpos)  # [B,H,Lq,T]
    H = sub.shape[1]
    out = {}
    for h1 in range(H):
        for h2 in range(h1 + 1, H):
            js = _js_divergence(sub[:, h1, :, :], sub[:, h2, :, :]).mean().item()
            out[f"attn/head_pair_{h1}_{h2}/js_full"] = float(js)
    return out


def _offset_mass_vectors(
    attn: torch.Tensor,
    query_token_positions: List[int],
    key_token_positions: List[int],
    max_abs_offset: int,
) -> torch.Tensor:
    """Return [H, num_offsets] mean per-query x-cell attention masses."""
    device = attn.device
    T = attn.shape[-1]
    qpos = torch.tensor([p for p in query_token_positions if p < T], device=device, dtype=torch.long)
    kpos = torch.tensor([p for p in key_token_positions if p < T], device=device, dtype=torch.long)
    sub = attn.index_select(2, qpos).index_select(3, kpos)  # [B,H,Lq,Lk]
    offsets = _offsets_for_positions(list(range(len(qpos))), list(range(len(kpos)))).to(device)
    vals = []
    for off in range(-max_abs_offset, max_abs_offset + 1):
        mask = offsets == off
        valid_queries = mask.any(dim=-1)
        if not valid_queries.any():
            vals.append(torch.zeros(sub.shape[1], device=device))
        else:
            mass = (sub * mask.float().view(1, 1, *mask.shape)).sum(dim=-1)  # [B,H,Lq]
            vals.append(mass[:, :, valid_queries].mean(dim=(0, 2)))  # [H]
    return torch.stack(vals, dim=-1)  # [H,O]


def head_offset_divergence(
    attn: torch.Tensor,
    query_token_positions: List[int],
    key_token_positions: List[int],
    max_abs_offset: int = 8,
) -> Dict[str, float]:
    """Pairwise divergence between heads' offset-mass profiles."""
    vec = _offset_mass_vectors(attn, query_token_positions, key_token_positions, max_abs_offset=max_abs_offset)  # [H,O]
    H = vec.shape[0]
    out = {}
    for h in range(H):
        top_idx = int(vec[h].argmax().item())
        out[f"attn/head_{h}/top_offset"] = float(top_idx - max_abs_offset)
    for h1 in range(H):
        for h2 in range(h1 + 1, H):
            v1 = vec[h1]
            v2 = vec[h2]
            cos = F.cosine_similarity(v1.view(1, -1), v2.view(1, -1), dim=-1).item()
            l1 = (v1 - v2).abs().mean().item()
            out[f"attn/head_pair_{h1}_{h2}/offset_cosine_distance"] = float(1.0 - cos)
            out[f"attn/head_pair_{h1}_{h2}/offset_l1_distance"] = float(l1)
    return out


def qk_scores_by_cell_offset(
    q: torch.Tensor,
    k: torch.Tensor,
    query_token_positions: List[int],
    key_token_positions: List[int],
    max_abs_offset: int = 8,
) -> Dict[str, float]:
    """Raw QK dot-product score by cell offset.

    q/k: [B,H,T,Dh].
    """
    device = q.device
    T = q.shape[2]
    qpos = torch.tensor([p for p in query_token_positions if p < T], device=device, dtype=torch.long)
    kpos = torch.tensor([p for p in key_token_positions if p < T], device=device, dtype=torch.long)
    qsub = q.index_select(2, qpos)
    ksub = k.index_select(2, kpos)
    scores = torch.matmul(qsub, ksub.transpose(-2, -1)) / (q.shape[-1] ** 0.5)  # [B,H,L,L]
    offsets = _offsets_for_positions(list(range(len(qpos))), list(range(len(kpos)))).to(device)
    out = {}
    for h in range(scores.shape[1]):
        for off in range(-max_abs_offset, max_abs_offset + 1):
            mask = offsets == off
            if mask.any():
                val = scores[:, h, :, :][..., mask].mean().item()
            else:
                val = 0.0
            out[f"qk/head_{h}/offset_{off}_score"] = float(val)
    return out


def qk_true_false_margin(
    q: torch.Tensor,
    k: torch.Tensor,
    query_token_positions: List[int],
    key_token_positions: List[int],
    true_offsets: List[int],
) -> Dict[str, float]:
    device = q.device
    T = q.shape[2]
    qpos = torch.tensor([p for p in query_token_positions if p < T], device=device, dtype=torch.long)
    kpos = torch.tensor([p for p in key_token_positions if p < T], device=device, dtype=torch.long)
    qsub = q.index_select(2, qpos)
    ksub = k.index_select(2, kpos)
    scores = torch.matmul(qsub, ksub.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
    offsets = _offsets_for_positions(list(range(len(qpos))), list(range(len(kpos)))).to(device)
    true_mask = torch.zeros_like(offsets, dtype=torch.bool)
    for off in true_offsets:
        true_mask |= offsets == off
    false_mask = ~true_mask
    out = {}
    for h in range(scores.shape[1]):
        true_score = scores[:, h, :, :][..., true_mask].mean().item()
        false_score = scores[:, h, :, :][..., false_mask].mean().item()
        out[f"qk/head_{h}/true_score"] = float(true_score)
        out[f"qk/head_{h}/false_score"] = float(false_score)
        out[f"qk/head_{h}/true_false_margin"] = float(true_score - false_score)
    return out


def prefix_metrics(metrics: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}/{k}": v for k, v in metrics.items()}
