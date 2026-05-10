"""Tiny decoder-only transformer with explicit Q/K/V caches."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    max_seq_len: int = 128
    d_model: int = 128
    n_layers: int = 1
    n_heads: int = 2
    d_mlp: int = 256
    dropout: float = 0.0


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.d_head).transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        B, T, _ = x.shape
        q = self._split_heads(self.W_Q(x))
        k = self._split_heads(self.W_K(x))
        v = self._split_heads(self.W_V(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask.view(1, 1, T, T), torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn_drop = self.dropout(attn)
        head_out = torch.matmul(attn_drop, v)  # [B,H,T,Dh]
        merged = self._merge_heads(head_out)
        out = self.W_O(merged)
        if return_cache:
            return out, {
                "q": q,
                "k": k,
                "v": v,
                "scores": scores,
                "attn": attn,
                "head_out": head_out,
                "attn_out": out,
            }
        return out, None


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            nn.GELU(),
            nn.Linear(cfg.d_mlp, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, return_cache: bool = False):
        resid_pre = x
        attn_in = self.ln1(x)
        attn_out, attn_cache = self.attn(attn_in, return_cache=return_cache)
        x = x + attn_out
        resid_mid = x
        mlp_in = self.ln2(x)
        # Break out MLP internals for cache.
        pre = self.mlp[0](mlp_in)
        post_act = self.mlp[1](pre)
        mlp_out = self.mlp[2](post_act)
        mlp_out = self.mlp[3](mlp_out)
        x = x + mlp_out
        if return_cache:
            cache = dict(attn_cache)
            cache.update({
                "resid_pre": resid_pre,
                "attn_normed": attn_in,
                "resid_mid": resid_mid,
                "mlp_normed": mlp_in,
                "mlp_pre": pre,
                "mlp_post": post_act,
                "mlp_out": mlp_out,
                "resid_post": x,
            })
            return x, cache
        return x, None


class TinyTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, return_cache: bool = False):
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len={self.cfg.max_seq_len}")
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_embed(input_ids) + self.pos_embed(pos)
        x = self.drop(x)
        full_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        for idx, block in enumerate(self.blocks):
            x, cache = block(x, return_cache=return_cache)
            if return_cache:
                full_cache[f"layer_{idx}"] = cache
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if return_cache:
            full_cache["final_resid"] = x
            return logits, full_cache
        return logits, None


def build_model_from_config(config, vocab_size: int) -> TinyTransformer:
    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        max_seq_len=int(config["model"].get("max_seq_len", 128)),
        d_model=int(config["model"].get("d_model", 128)),
        n_layers=int(config["model"].get("n_layers", 1)),
        n_heads=int(config["model"].get("n_heads", 2)),
        d_mlp=int(config["model"].get("d_mlp", 256)),
        dropout=float(config["model"].get("dropout", 0.0)),
    )
    return TinyTransformer(model_cfg)
