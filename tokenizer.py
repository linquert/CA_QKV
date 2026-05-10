"""A deliberately small tokenizer for CA row-to-row experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass
class CATokenizer:
    max_row_length: int = 128
    include_position_tokens: bool = False

    def __post_init__(self):
        base_tokens = ["<PAD>", "<BOS>", "<EOS>", "<X>", "<Y>", "<SEP>", "0", "1"]
        pos_tokens = [f"p{i}" for i in range(self.max_row_length)] if self.include_position_tokens else []
        self.tokens: List[str] = base_tokens + pos_tokens
        self.stoi: Dict[str, int] = {tok: i for i, tok in enumerate(self.tokens)}
        self.itos: Dict[int, str] = {i: tok for tok, i in self.stoi.items()}

    @property
    def pad_id(self) -> int:
        return self.stoi["<PAD>"]

    @property
    def bos_id(self) -> int:
        return self.stoi["<BOS>"]

    @property
    def eos_id(self) -> int:
        return self.stoi["<EOS>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode_tokens(self, tokens: Sequence[str]) -> List[int]:
        try:
            return [self.stoi[t] for t in tokens]
        except KeyError as e:
            raise KeyError(f"Unknown token {e}; vocab={self.tokens}") from e

    def decode_ids(self, ids: Sequence[int]) -> List[str]:
        return [self.itos[int(i)] for i in ids]

    def bit_to_token(self, bit: int) -> str:
        return "1" if int(bit) == 1 else "0"

    def token_to_bit(self, tok: str) -> int:
        if tok not in {"0", "1"}:
            raise ValueError(f"Expected bit token, got {tok}")
        return int(tok)

    def build_sequence_tokens(self, x_bits: Sequence[int], y_bits: Sequence[int]) -> List[str]:
        if self.include_position_tokens:
            x_part = []
            y_part = []
            for i, bit in enumerate(x_bits):
                x_part.extend([f"p{i}", self.bit_to_token(bit)])
            for i, bit in enumerate(y_bits):
                y_part.extend([f"p{i}", self.bit_to_token(bit)])
            return ["<BOS>", "<X>"] + x_part + ["<SEP>", "<Y>"] + y_part + ["<EOS>"]
        return ["<BOS>", "<X>"] + [self.bit_to_token(b) for b in x_bits] + ["<SEP>", "<Y>"] + [self.bit_to_token(b) for b in y_bits] + ["<EOS>"]

    def encode_example(self, x_bits: Sequence[int], y_bits: Sequence[int], max_seq_len: int | None = None):
        tokens = self.build_sequence_tokens(x_bits, y_bits)
        ids = self.encode_tokens(tokens)
        if max_seq_len is not None and len(ids) > max_seq_len:
            raise ValueError(f"Sequence length {len(ids)} exceeds max_seq_len={max_seq_len}")
        return ids, tokens

    def y_token_positions(self, row_length: int) -> List[int]:
        """Positions in the full token sequence occupied by target y bit tokens.

        This is for the unshifted full sequence <BOS> <X> x... <SEP> <Y> y... <EOS>.
        """
        if self.include_position_tokens:
            # <BOS>, <X>, 2L x tokens, <SEP>, <Y>, then pairs p_i, y_i
            start = 2 + 2 * row_length + 2
            return [start + 2 * i + 1 for i in range(row_length)]
        start = 2 + row_length + 2
        return [start + i for i in range(row_length)]

    def x_token_positions(self, row_length: int) -> List[int]:
        if self.include_position_tokens:
            start = 2
            return [start + 2 * i + 1 for i in range(row_length)]
        start = 2
        return [start + i for i in range(row_length)]

    def pred_positions_for_y(self, row_length: int) -> List[int]:
        """Input positions whose logits predict y tokens under next-token LM loss.

        If y token is at full sequence position p, it is predicted by logits at p-1.
        """
        return [p - 1 for p in self.y_token_positions(row_length)]



    def structural_token_positions(self, row_length: int) -> Dict[str, List[int]]:
        """Return named structural token positions in the unshifted full sequence.

        Positions are compatible with model input positions except <EOS>, which
        is omitted from the model input by next-token shifting. Downstream metric
        code filters positions that are outside the current attention length.
        """
        if self.include_position_tokens:
            sep = 2 + 2 * row_length
            y_marker = sep + 1
            x_pos_tokens = [2 + 2 * i for i in range(row_length)]
            y_pos_tokens = [y_marker + 1 + 2 * i for i in range(row_length)]
            return {
                "bos": [0],
                "x_marker": [1],
                "sep": [sep],
                "y_marker": [y_marker],
                "x_position_tokens": x_pos_tokens,
                "y_position_tokens": y_pos_tokens,
                "eos": [y_marker + 1 + 2 * row_length],
            }
        sep = 2 + row_length
        y_marker = sep + 1
        return {
            "bos": [0],
            "x_marker": [1],
            "sep": [sep],
            "y_marker": [y_marker],
            "eos": [y_marker + 1 + row_length],
        }

    def attention_token_groups(self, row_length: int) -> Dict[str, List[int]]:
        """Token-position groups for attention-sink diagnostics.

        The groups are intentionally allowed to overlap. For example,
        `structural` is a union of marker tokens, while `bos`, `sep`, etc. are
        also logged separately. `previous_y_bits` includes all y-bit positions;
        causal masking makes future y bits receive zero attention for a given
        query, so this remains safe and simple.
        """
        structural = self.structural_token_positions(row_length)
        x_cells = self.x_token_positions(row_length)
        y_bits = self.y_token_positions(row_length)
        marker_keys = ["bos", "x_marker", "sep", "y_marker"]
        structural_positions: List[int] = []
        for key in marker_keys:
            structural_positions.extend(structural.get(key, []))
        if self.include_position_tokens:
            # Position tokens are not cell-state payloads. Track them separately
            # and include them in a broader non-cell structural bucket.
            structural_positions.extend(structural.get("x_position_tokens", []))
            structural_positions.extend(structural.get("y_position_tokens", []))
        return {
            "bos": structural.get("bos", []),
            "x_marker": structural.get("x_marker", []),
            "sep": structural.get("sep", []),
            "y_marker": structural.get("y_marker", []),
            "structural": sorted(set(structural_positions)),
            "x_cells": x_cells,
            "previous_y_bits": y_bits,
        }

    def extract_predicted_y_bits_from_logits(self, logits, row_length: int):
        """Return predicted y bits from logits [B, T, V] at y prediction positions."""
        import torch
        pred_positions = self.pred_positions_for_y(row_length)
        bit_token_ids = torch.tensor([self.stoi["0"], self.stoi["1"]], device=logits.device)
        bit_logits = logits[:, pred_positions, :].index_select(-1, bit_token_ids)
        return bit_logits.argmax(dim=-1)  # [B, L], values 0/1 because bit_token_ids ordered 0,1
