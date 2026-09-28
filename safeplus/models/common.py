from __future__ import annotations

import torch
from torch import nn


class SequenceEncoder(nn.Module):
    """Shared unidirectional GRU: x[B,L,D] -> hidden states [B,L,H]."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_h, _ = self.gru(packed)

        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_h,
            batch_first=True,
            total_length=x.shape[1],
        )

        return h


class LSTMSequenceEncoder(nn.Module):
    """Shared unidirectional LSTM: x[B,L,D] -> hidden states [B,L,H]."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_h, _ = self.lstm(packed)

        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_h,
            batch_first=True,
            total_length=x.shape[1],
        )

        return h


def time_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return (
        torch.arange(max_len, device=lengths.device)[None, :]
        < lengths[:, None]
    )