"""Feature scaling that cannot see the future.

The defect this module exists to prevent: calling ``fit_transform`` on a whole price series
and *then* splitting it. The scaler's minimum and maximum are the lowest and highest close in
the entire period, including the held-out one, so every training row is normalised using a
constant derived from data the model is not supposed to have. The effect is subtle — the
model never sees a test *row* — but the test period's range is baked into the input
distribution, and reported error drops accordingly.

:class:`TrainOnlyScaler` makes that mistake awkward rather than easy: it fits once, refuses to
be refitted, and refuses to transform before it has been fitted.
"""

from __future__ import annotations

import numpy as np


class NotFittedError(RuntimeError):
    """Raised when a scaler is used before it has been fitted."""


class AlreadyFittedError(RuntimeError):
    """Raised on a second call to ``fit``.

    Refitting is almost always the leak: a pipeline that fits on train, then fits again on
    the full frame before predicting, looks correct line by line and is not.
    """


class TrainOnlyScaler:
    """Min-max scaler to [0, 1] whose statistics come from one array and never change.

    Values outside the fitted range map outside [0, 1] rather than being clipped. That is
    deliberate: a test-period price above anything seen in training *should* land above 1.0,
    and silently clipping it would hide exactly the distribution shift worth knowing about.
    """

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None
        self._n_fit_rows: int | None = None

    @property
    def fitted(self) -> bool:
        return self.min_ is not None

    @property
    def n_fit_rows(self) -> int:
        """How many rows the statistics were computed from. Recorded so it can be asserted."""
        if self._n_fit_rows is None:
            raise NotFittedError("scaler has not been fitted")
        return self._n_fit_rows

    def fit(self, x) -> TrainOnlyScaler:
        if self.fitted:
            raise AlreadyFittedError(
                "this scaler is already fitted; create a new one rather than refitting"
            )
        x = np.asarray(x, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {x.shape}")
        if x.shape[0] == 0:
            raise ValueError("cannot fit on an empty array")

        self.min_ = np.nanmin(x, axis=0)
        self.max_ = np.nanmax(x, axis=0)
        self._n_fit_rows = int(x.shape[0])
        return self

    def transform(self, x) -> np.ndarray:
        if not self.fitted:
            raise NotFittedError("call fit() on the training partition first")
        x = np.asarray(x, dtype=float)
        span = self.max_ - self.min_
        # A constant column has zero span; map it to zero instead of dividing by zero.
        span = np.where(span == 0, 1.0, span)
        return (x - self.min_) / span

    def inverse_transform(self, x) -> np.ndarray:
        """Map scaled values back to dollars. Every reported metric goes through this."""
        if not self.fitted:
            raise NotFittedError("call fit() on the training partition first")
        x = np.asarray(x, dtype=float)
        span = self.max_ - self.min_
        span = np.where(span == 0, 1.0, span)
        return x * span + self.min_


def fit_on_train(train_x) -> TrainOnlyScaler:
    """The only sanctioned way to build a scaler: from the training partition, once."""
    return TrainOnlyScaler().fit(train_x)
