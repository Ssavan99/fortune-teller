"""Chronological train/validation/test splits.

Splits are defined by **fixed calendar cutoffs**, not by fractions of the dataset. A ratio
split silently moves every boundary when the data is extended, which makes two runs of the
same code incomparable. Cutoffs do not move.

The test period begins 2024-03-01 and runs to the end of the snapshot: two and a half years
that no model in this repository sees during fitting, scaling, early stopping or model
selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TRAIN_END = pd.Timestamp("2023-06-30")
VAL_END = pd.Timestamp("2024-02-29")

#: Experiment B stops here. The sentiment snapshot itself runs to 2024-03-25, which reaches
#: 25 days into the main held-out period — so it is clamped to ``VAL_END`` instead. Selecting
#: an architecture on a window that overlapped the held-out period would compromise the
#: headline result, for the sake of 17 extra days of ablation data.
SENTIMENT_END = VAL_END
SENTIMENT_TRAIN_END = pd.Timestamp("2023-03-31")
SENTIMENT_VAL_END = pd.Timestamp("2023-09-30")

assert SENTIMENT_END <= VAL_END, "Experiment B must not reach into the main held-out period"
assert SENTIMENT_TRAIN_END < SENTIMENT_VAL_END < SENTIMENT_END
assert TRAIN_END < VAL_END


@dataclass(frozen=True)
class Split:
    """One chronological partition of a long-format frame."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        """Assert the partitions are non-empty and strictly ordered in time.

        This is the guard that makes 'no lookahead' a property of the code rather than a
        claim in a README.
        """
        for name, part in (("train", self.train), ("val", self.val), ("test", self.test)):
            if part.empty:
                raise ValueError(f"{name} split is empty")

        if self.train["date"].max() >= self.val["date"].min():
            raise ValueError("train overlaps val")
        if self.val["date"].max() >= self.test["date"].min():
            raise ValueError("val overlaps test")

    def describe(self) -> str:
        rows = []
        for name, part in (("train", self.train), ("val", self.val), ("test", self.test)):
            rows.append(
                f"  {name:5s} {len(part):6,d} rows  "
                f"{part['date'].min().date()} -> {part['date'].max().date()}"
            )
        return "\n".join(rows)


def chronological_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp = TRAIN_END,
    val_end: pd.Timestamp = VAL_END,
) -> Split:
    """Partition a long-format frame on ``date`` at two fixed cutoffs.

    Boundaries are inclusive of ``train_end`` and ``val_end`` respectively.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    date = df["date"]
    return Split(
        train=df[date <= train_end].copy(),
        val=df[(date > train_end) & (date <= val_end)].copy(),
        test=df[date > val_end].copy(),
    )


def sentiment_window_split(df: pd.DataFrame) -> Split:
    """Split for Experiment B, confined to the window where sentiment exists."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    within = df[df["date"] <= SENTIMENT_END].copy()
    return chronological_split(
        within, train_end=SENTIMENT_TRAIN_END, val_end=SENTIMENT_VAL_END
    )
