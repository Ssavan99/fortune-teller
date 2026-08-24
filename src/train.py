"""Training loop with early stopping on validation loss.

Model selection touches the validation partition only. The test partition is loaded once, at
the very end, to produce the reported numbers — never to choose an epoch, an architecture or a
learning rate.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm import BiLSTMRegressor, count_parameters
from src.sequences import SequenceSet


@dataclass
class TrainConfig:
    """Everything that changes a result. Serialised alongside the metrics."""

    target: str = "level"
    lookback: int = 20
    hidden: int = 100
    layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 128
    max_epochs: int = 60
    patience: int = 8
    seed: int = 20260822
    history: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.pop("history", None)
        return d


class TargetNormalizer:
    """Standardise the regression target using training statistics only.

    Raw daily returns sit around ±0.02, which makes for very small losses and awkward
    learning rates. Standardising is a numerical convenience, not a modelling choice, and it
    is inverted before anything is converted to dollars.
    """

    def __init__(self, y_train: np.ndarray) -> None:
        self.mean = float(np.mean(y_train))
        self.std = float(np.std(y_train))
        if self.std == 0:
            self.std = 1.0

    def forward(self, y: np.ndarray) -> np.ndarray:
        return (np.asarray(y, dtype=float) - self.mean) / self.std

    def inverse(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float) * self.std + self.mean


def set_seed(seed: int) -> None:
    """Seed every source of randomness the run touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loader(seqs: SequenceSet, norm: TargetNormalizer, batch_size: int, shuffle: bool):
    x = torch.tensor(seqs.x, dtype=torch.float32)
    y = torch.tensor(norm.forward(seqs.y), dtype=torch.float32)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn) -> float:
    model.eval()
    total, n = 0.0, 0
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        total += loss.item() * len(yb)
        n += len(yb)
    return total / max(n, 1)


def train(
    train_seqs: SequenceSet,
    val_seqs: SequenceSet,
    config: TrainConfig,
    verbose: bool = True,
) -> tuple[nn.Module, TargetNormalizer, TrainConfig]:
    """Fit a model, keeping the weights from the best validation epoch."""
    set_seed(config.seed)

    norm = TargetNormalizer(train_seqs.y)
    train_loader = _loader(train_seqs, norm, config.batch_size, shuffle=True)
    val_loader = _loader(val_seqs, norm, config.batch_size, shuffle=False)

    model = BiLSTMRegressor(
        n_features=train_seqs.x.shape[2],
        hidden=config.hidden,
        layers=config.layers,
        dropout=config.dropout,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    if verbose:
        print(
            f"  {count_parameters(model):,} parameters | "
            f"{len(train_seqs):,} train / {len(val_seqs):,} val sequences"
        )

    best_loss = float("inf")
    best_state = None
    since_improved = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            optimiser.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running += loss.item() * len(yb)
            n += len(yb)

        train_loss = running / max(n, 1)
        val_loss = _evaluate_loss(model, val_loader, loss_fn)
        config.history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
        else:
            since_improved += 1

        if verbose and (epoch == 1 or epoch % 5 == 0):
            print(f"  epoch {epoch:3d}  train {train_loss:.5f}  val {val_loss:.5f}")

        if since_improved >= config.patience:
            if verbose:
                print(f"  early stop at epoch {epoch} (best val {best_loss:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, norm, config


@torch.no_grad()
def predict(model: nn.Module, seqs: SequenceSet, norm: TargetNormalizer) -> np.ndarray:
    """Predictions in target space — scaled level, or raw return."""
    model.eval()
    x = torch.tensor(seqs.x, dtype=torch.float32)
    out = []
    for i in range(0, len(x), 512):
        out.append(model(x[i : i + 512]).numpy())
    return norm.inverse(np.concatenate(out))
