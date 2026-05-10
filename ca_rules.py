"""Cellular automaton rule definitions.

The first experimental suite uses 1D binary radius-1 rules.
Each local neighborhood is represented as a tuple: (left, center, right).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

Neighborhood = Tuple[int, int, int]


@dataclass(frozen=True)
class CARule:
    """Base class for radius-1 binary cellular automaton rules."""

    name: str
    radius: int = 1
    num_states: int = 2

    def apply_local(self, neighborhood: Neighborhood) -> int:
        raise NotImplementedError

    @property
    def local_table(self) -> Dict[Neighborhood, int]:
        return {nb: int(self.apply_local(nb)) for nb in all_binary_radius1_neighborhoods()}

    def neighborhood_to_id(self, neighborhood: Sequence[int]) -> int:
        """Map (l,c,r) to id 0..7 using binary value l*4+c*2+r."""
        l, c, r = [int(x) for x in neighborhood]
        return 4 * l + 2 * c + r

    def id_to_neighborhood(self, idx: int) -> Neighborhood:
        idx = int(idx)
        return ((idx >> 2) & 1, (idx >> 1) & 1, idx & 1)

    def neighborhood_label(self, idx_or_nb) -> str:
        if isinstance(idx_or_nb, int):
            nb = self.id_to_neighborhood(idx_or_nb)
        else:
            nb = tuple(int(x) for x in idx_or_nb)
        return "".join(str(x) for x in nb)

    def pad_value(self, boundary_condition: str, side: str, row: torch.Tensor) -> int:
        if boundary_condition == "zero":
            return 0
        if boundary_condition == "one":
            return 1
        if boundary_condition == "periodic":
            return int(row[-1].item()) if side == "left" else int(row[0].item())
        if boundary_condition == "reflect":
            return int(row[0].item()) if side == "left" else int(row[-1].item())
        raise ValueError(f"Unknown boundary_condition: {boundary_condition}")

    def get_neighborhoods(self, row: torch.Tensor, boundary_condition: str = "zero") -> torch.Tensor:
        """Return [L, 3] neighborhoods for a row tensor [L]."""
        if row.ndim != 1:
            raise ValueError(f"Expected row shape [L], got {tuple(row.shape)}")
        L = row.shape[0]
        out = []
        for i in range(L):
            left = int(row[i - 1].item()) if i > 0 else self.pad_value(boundary_condition, "left", row)
            center = int(row[i].item())
            right = int(row[i + 1].item()) if i < L - 1 else self.pad_value(boundary_condition, "right", row)
            out.append([left, center, right])
        return torch.tensor(out, dtype=torch.long)

    def apply_row(self, row: torch.Tensor, boundary_condition: str = "zero") -> torch.Tensor:
        neighborhoods = self.get_neighborhoods(row, boundary_condition=boundary_condition)
        vals = [self.apply_local(tuple(int(x) for x in nb.tolist())) for nb in neighborhoods]
        return torch.tensor(vals, dtype=torch.long)

    def neighborhood_ids(self, row: torch.Tensor, boundary_condition: str = "zero") -> torch.Tensor:
        neighborhoods = self.get_neighborhoods(row, boundary_condition=boundary_condition)
        ids = [self.neighborhood_to_id(nb.tolist()) for nb in neighborhoods]
        return torch.tensor(ids, dtype=torch.long)


class LeftCopyRule(CARule):
    def __init__(self):
        super().__init__(name="LEFT_COPY")

    def apply_local(self, neighborhood: Neighborhood) -> int:
        return int(neighborhood[0])


class CenterCopyRule(CARule):
    def __init__(self):
        super().__init__(name="CENTER_COPY")

    def apply_local(self, neighborhood: Neighborhood) -> int:
        return int(neighborhood[1])


class RightCopyRule(CARule):
    def __init__(self):
        super().__init__(name="RIGHT_COPY")

    def apply_local(self, neighborhood: Neighborhood) -> int:
        return int(neighborhood[2])


class MajorityRule(CARule):
    def __init__(self):
        super().__init__(name="MAJORITY")

    def apply_local(self, neighborhood: Neighborhood) -> int:
        return int(sum(neighborhood) >= 2)


class XORRule(CARule):
    def __init__(self):
        super().__init__(name="XOR")

    def apply_local(self, neighborhood: Neighborhood) -> int:
        return int(neighborhood[0] ^ neighborhood[1] ^ neighborhood[2])


class ElementaryRule(CARule):
    """Elementary CA rule by Wolfram number, e.g. 110 or 30.

    Wolfram convention orders neighborhoods as:
        111, 110, 101, 100, 011, 010, 001, 000
    The binary digits of rule_number give outputs in that order.
    """

    def __init__(self, rule_number: int):
        if not (0 <= int(rule_number) <= 255):
            raise ValueError("Elementary rule number must be in [0, 255]")
        self.rule_number = int(rule_number)
        bits = f"{self.rule_number:08b}"
        ordered = [(1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
                   (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0)]
        self._table = {nb: int(bit) for nb, bit in zip(ordered, bits)}
        super().__init__(name=f"RULE_{self.rule_number}")

    @property
    def local_table(self) -> Dict[Neighborhood, int]:
        return dict(self._table)

    def apply_local(self, neighborhood: Neighborhood) -> int:
        return int(self._table[tuple(int(x) for x in neighborhood)])


def all_binary_radius1_neighborhoods() -> List[Neighborhood]:
    return [(a, b, c) for a, b, c in product([0, 1], repeat=3)]


def build_rule(rule_cfg) -> CARule:
    """Build a rule from a config namespace/dict.

    Accepts names: LEFT_COPY, CENTER_COPY, RIGHT_COPY, MAJORITY, XOR, RULE_110, RULE_30.
    For arbitrary elementary rules, use name: ELEMENTARY and rule_number: N.
    """
    if isinstance(rule_cfg, dict):
        name = str(rule_cfg.get("name", "LEFT_COPY")).upper()
        rule_number = rule_cfg.get("rule_number", None)
    else:
        name = str(getattr(rule_cfg, "name", "LEFT_COPY")).upper()
        rule_number = getattr(rule_cfg, "rule_number", None)

    if name == "LEFT_COPY":
        return LeftCopyRule()
    if name == "CENTER_COPY":
        return CenterCopyRule()
    if name == "RIGHT_COPY":
        return RightCopyRule()
    if name == "MAJORITY":
        return MajorityRule()
    if name == "XOR":
        return XORRule()
    if name.startswith("RULE_"):
        return ElementaryRule(int(name.split("_", 1)[1]))
    if name == "ELEMENTARY":
        if rule_number is None:
            raise ValueError("ELEMENTARY rule requires rule_number")
        return ElementaryRule(int(rule_number))
    raise ValueError(f"Unknown CA rule: {name}")


def true_offsets_for_rule(rule: CARule) -> List[int]:
    """Return expected important relative offsets for first-order analysis."""
    if rule.name == "LEFT_COPY":
        return [-1]
    if rule.name == "CENTER_COPY":
        return [0]
    if rule.name == "RIGHT_COPY":
        return [1]
    return [-1, 0, 1]
