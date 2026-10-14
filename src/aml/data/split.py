from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIODS = ("warmup", "train", "val", "threshold", "test", "tail")
MODEL_PERIODS = ("train", "val", "threshold", "test")


@dataclass(frozen=True)
class SplitConfig:
    unit: str = "1D"
    warmup_units: int = 1
    val_units: int = 1
    threshold_units: int = 1
    test_size: float = 0.15
    # Trailing units with fewer rows than this share of the median unit are excluded.
    # None keeps the tail.
    tail_min_share: float | None = 0.01

    def __post_init__(self):
        if pd.Timedelta(self.unit).total_seconds() <= 0:
            raise ValueError("unit must be a positive duration")
        if self.warmup_units < 0 or self.val_units < 1 or self.threshold_units < 1:
            raise ValueError("val and threshold need at least one unit; warmup cannot be negative")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be in (0, 1)")
        if self.tail_min_share is not None and not 0 < self.tail_min_share < 1:
            raise ValueError("tail_min_share must be in (0, 1)")


@dataclass(frozen=True)
class SplitBoundaries:
    warmup: slice
    train: slice
    val: slice
    threshold: slice
    test: slice
    tail: slice
    # Inclusive start timestamp of each period; rows are assigned by searchsorted.
    starts: dict[str, pd.Timestamp]

    def slices(self) -> dict[str, slice]:
        return {name: getattr(self, name) for name in PERIODS}


def temporal_slices(timestamps: pd.Series, config: SplitConfig = SplitConfig()) -> SplitBoundaries:
    timestamps = _sorted_timestamps(timestamps)
    unit = pd.Timedelta(config.unit)

    kept = _tail_start(timestamps, config)
    if kept == 0:
        raise ValueError("Every period is sparse: nothing left after removing the tail")

    test_start = timestamps.iloc[int(kept * (1 - config.test_size))].floor(unit)
    threshold_start = test_start - config.threshold_units * unit
    val_start = threshold_start - config.val_units * unit
    train_start = timestamps.iloc[0].floor(unit) + config.warmup_units * unit
    if not train_start < val_start:
        raise ValueError("No training units left between warmup and validation")

    starts = {
        "warmup": timestamps.iloc[0],
        "train": train_start,
        "val": val_start,
        "threshold": threshold_start,
        "test": test_start,
    }
    if kept < len(timestamps):
        starts["tail"] = timestamps.iloc[kept]
    boundaries = boundaries_from_starts(timestamps, starts)
    validate_split_boundaries(timestamps, boundaries)
    return boundaries


def boundaries_from_starts(timestamps: pd.Series, starts: dict) -> SplitBoundaries:
    """Rebuild period slices from recorded start timestamps (e.g. from a run manifest)."""
    timestamps = _sorted_timestamps(timestamps)
    n = len(timestamps)
    starts = {name: pd.Timestamp(value) for name, value in starts.items() if value is not None}
    edges = [0] + [
        int(timestamps.searchsorted(starts[name], side="left")) if name in starts else n
        for name in PERIODS[1:]
    ] + [n]
    if any(left > right for left, right in zip(edges, edges[1:])):
        raise ValueError("Period starts must be chronological")
    slices = {name: slice(left, right) for name, left, right in zip(PERIODS, edges, edges[1:])}
    return SplitBoundaries(**slices, starts=starts)


def validate_split_boundaries(timestamps: pd.Series, boundaries: SplitBoundaries) -> dict:
    """Check contiguous coverage and strict temporal order; return a per-period report."""
    timestamps = _sorted_timestamps(timestamps)
    previous_stop, previous_end = 0, None
    report = {}
    for name, part in boundaries.slices().items():
        if part.step not in (None, 1) or part.start != previous_stop or part.stop < part.start:
            raise ValueError(f"{name}: periods must be contiguous positional slices")
        previous_stop = part.stop
        values = timestamps.iloc[part]
        if values.empty:
            if name in MODEL_PERIODS:
                raise ValueError(f"{name} is empty")
            report[name] = {"rows": 0, "min_timestamp": None, "max_timestamp": None}
            continue
        if previous_end is not None and previous_end >= values.iloc[0]:
            raise ValueError(f"Temporal overlap before {name}")
        previous_end = values.iloc[-1]
        report[name] = {
            "rows": len(values),
            "min_timestamp": values.iloc[0].isoformat(),
            "max_timestamp": values.iloc[-1].isoformat(),
        }
    if previous_stop != len(timestamps):
        raise ValueError("Periods do not cover all observations")
    return report


def _tail_start(timestamps: pd.Series, config: SplitConfig) -> int:
    if config.tail_min_share is None:
        return len(timestamps)
    units = timestamps.dt.floor(config.unit)
    counts = units.value_counts(sort=False).sort_index()
    sparse = counts.to_numpy() < config.tail_min_share * np.median(counts.to_numpy())
    dense = np.flatnonzero(~sparse)
    if not len(dense):
        return 0
    if dense[-1] == len(counts) - 1:
        return len(timestamps)
    return int(units.searchsorted(counts.index[dense[-1] + 1], side="left"))


def _sorted_timestamps(timestamps) -> pd.Series:
    timestamps = pd.to_datetime(pd.Series(timestamps), errors="raise").reset_index(drop=True)
    if timestamps.empty or timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Expected nonempty, sorted timestamps without NaT")
    return timestamps

