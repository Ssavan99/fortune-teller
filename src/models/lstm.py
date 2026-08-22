"""A bidirectional LSTM regressor.

Two stacked bidirectional layers of 100 units, matching the size commonly used for this task,
followed by a linear head. The final hidden state of both directions is concatenated and fed
to the head.

A note on "bidirectional": it is standard in this literature and is kept here for
comparability, but it does **not** let the model see the future. The sequence handed to the
model ends the day before the target, so the backward pass runs over the same past window in
reverse. No information from the target day or later is in the tensor at all — that property
is enforced in :mod:`src.sequences` and asserted in the tests.
"""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMRegressor(nn.Module):
    """Predict a single scalar from a window of features."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 100,
        layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        hidden = self.lstm.hidden_size

        # The two directions finish at opposite ends of the sequence. PyTorch packs the
        # forward state in the first `hidden` channels and the backward state in the rest,
        # both indexed by timestep — so the forward direction's final state is at t = -1 and
        # the backward direction's final state is at t = 0.
        #
        # Taking out[:, -1, :] for both, which reads naturally, silently feeds the head a
        # backward state that has consumed only the last timestep and nothing else.
        forward_last = out[:, -1, :hidden]
        backward_last = out[:, 0, hidden:]
        return self.head(torch.cat([forward_last, backward_last], dim=1)).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
