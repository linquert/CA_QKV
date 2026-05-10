"""Synthetic CA datasets with controlled distribution knobs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .ca_rules import CARule
from .tokenizer import CATokenizer


@dataclass
class CADatasetConfig:
    row_length: int = 16
    num_examples: int = 10000
    distribution: str = "random"
    boundary_condition: str = "zero"
    max_seq_len: int = 128
    single_neighborhood_id: Optional[int] = None
    smooth_flip_prob: float = 0.08


class CADataset(Dataset):
    """Pre-generated CA examples.

    The dataset stores metadata needed for per-neighborhood, per-position, and
    QKV diagnostics. First implementation focuses on full-row prediction.
    """

    def __init__(
        self,
        rule: CARule,
        tokenizer: CATokenizer,
        cfg: CADatasetConfig,
        split: str = "train",
        seed: int = 0,
    ):
        self.rule = rule
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.split = split
        self.generator = torch.Generator().manual_seed(int(seed))
        self.examples = [self._make_example(i) for i in range(cfg.num_examples)]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.examples[idx]

    def _make_row_random(self) -> torch.Tensor:
        return torch.randint(0, 2, (self.cfg.row_length,), generator=self.generator, dtype=torch.long)

    def _make_row_smooth(self) -> torch.Tensor:
        L = self.cfg.row_length
        row = torch.empty(L, dtype=torch.long)
        cur = int(torch.randint(0, 2, (), generator=self.generator).item())
        for i in range(L):
            if i > 0 and torch.rand((), generator=self.generator).item() < self.cfg.smooth_flip_prob:
                cur = 1 - cur
            row[i] = cur
        return row

    def _make_row_boundary_heavy(self) -> torch.Tensor:
        L = self.cfg.row_length
        choice = int(torch.randint(0, 3, (), generator=self.generator).item())
        if choice == 0:
            return torch.tensor([(i % 2) for i in range(L)], dtype=torch.long)
        if choice == 1:
            return torch.tensor([((i // 2) % 2) for i in range(L)], dtype=torch.long)
        return self._make_row_random()

    def _make_row_single_neighborhood(self) -> torch.Tensor:
        """Construct rows containing a requested neighborhood somewhere.

        This is a lightweight first version: it seeds a random row and implants
        the desired triple at a random interior position. Full balancing across
        all positions can be added later.
        """
        row = self._make_row_random()
        nb_id = self.cfg.single_neighborhood_id
        if nb_id is None:
            nb_id = int(torch.randint(0, 8, (), generator=self.generator).item())
        nb = self.rule.id_to_neighborhood(nb_id)
        if self.cfg.row_length >= 3:
            i = int(torch.randint(1, self.cfg.row_length - 1, (), generator=self.generator).item())
            row[i - 1:i + 2] = torch.tensor(nb, dtype=torch.long)
        return row

    def _make_row_balanced(self, idx: int) -> torch.Tensor:
        """Approximate full-row neighborhood balancing.

        For full-row prediction exact balancing is nontrivial. This version cycles
        through target neighborhood IDs and implants them. Across many examples,
        each neighborhood receives targeted exposure, while rows remain natural.
        Use distribution=single_cell_balanced when exact supervised-neighborhood
        balance is needed.
        """
        row = self._make_row_random()
        target_id = idx % 8
        nb = self.rule.id_to_neighborhood(target_id)
        if self.cfg.row_length >= 3:
            i = int(torch.randint(1, self.cfg.row_length - 1, (), generator=self.generator).item())
            row[i - 1:i + 2] = torch.tensor(nb, dtype=torch.long)
        return row

    def _make_row_single_cell_balanced(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Exactly balance the *supervised* cell over the 8 neighborhoods.

        The sequence still contains the full next row, but the loss mask supervises
        only one output cell. The queried cell's local neighborhood is exactly
        `idx % 8`; query positions cycle across interior cells. This is the clean
        dataset mode for per-update and per-neighborhood causal analysis because
        the training signal is attributable to a known local triple.
        """
        L = self.cfg.row_length
        if L < 3:
            raise ValueError("single_cell_balanced requires row_length >= 3 for radius-1 CA")
        row = self._make_row_random()
        target_id = idx % 8
        interior_count = L - 2
        query_pos = 1 + ((idx // 8) % interior_count)
        nb = self.rule.id_to_neighborhood(target_id)
        row[query_pos - 1:query_pos + 2] = torch.tensor(nb, dtype=torch.long)
        return row, query_pos

    def _sample_row(self, idx: int) -> torch.Tensor:
        dist = self.cfg.distribution.lower()
        if dist in {"random", "uniform"}:
            return self._make_row_random()
        if dist in {"balanced", "balanced_neighborhoods"}:
            return self._make_row_balanced(idx)
        if dist in {"skewed", "skewed_smooth", "smooth"}:
            return self._make_row_smooth()
        if dist in {"boundary_heavy", "alternating"}:
            return self._make_row_boundary_heavy()
        if dist in {"single_neighborhood", "rare_neighborhood"}:
            return self._make_row_single_neighborhood()
        raise ValueError(f"Unknown distribution: {self.cfg.distribution}")

    def _make_example(self, idx: int) -> Dict[str, torch.Tensor]:
        dist = self.cfg.distribution.lower()
        query_mask = torch.ones(self.cfg.row_length, dtype=torch.float32)
        if dist in {"single_cell_balanced", "balanced_single_cell", "exact_balanced", "exact_single_cell_balanced"}:
            x, query_pos = self._make_row_single_cell_balanced(idx)
            query_mask.zero_()
            query_mask[query_pos] = 1.0
        else:
            x = self._sample_row(idx)
        y = self.rule.apply_row(x, boundary_condition=self.cfg.boundary_condition)
        neighborhoods = self.rule.get_neighborhoods(x, boundary_condition=self.cfg.boundary_condition)
        neighborhood_ids = self.rule.neighborhood_ids(x, boundary_condition=self.cfg.boundary_condition)
        input_ids_full, tokens = self.tokenizer.encode_example(x.tolist(), y.tolist(), max_seq_len=self.cfg.max_seq_len)

        input_ids_full = torch.tensor(input_ids_full, dtype=torch.long)
        # Next-token LM. Feed all but last token; target is all but first token.
        input_ids = input_ids_full[:-1]
        target_ids = input_ids_full[1:]

        # Loss only on supervised y bit tokens. If full y token is at position p,
        # target_ids index p-1. Full-row modes supervise all cells; exact
        # single-cell modes supervise only the selected query cell.
        loss_mask = torch.zeros_like(target_ids, dtype=torch.float32)
        for cell_idx, full_pos in enumerate(self.tokenizer.y_token_positions(self.cfg.row_length)):
            target_idx = full_pos - 1
            loss_mask[target_idx] = query_mask[cell_idx]

        return {
            "input_ids": input_ids,
            "target_ids": target_ids,
            "loss_mask": loss_mask,
            "row": x,
            "next_row": y,
            "neighborhoods": neighborhoods,
            "neighborhood_ids": neighborhood_ids,
            "positions": torch.arange(self.cfg.row_length, dtype=torch.long),
            "query_mask": query_mask,
        }


def collate_ca_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        else:
            out[k] = vals
    return out
