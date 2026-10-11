from __future__ import annotations

import numpy as np
import pandas as pd

from dataclasses import dataclass

@dataclass(frozen=True)
class SplitBoundaries:
    train: slice
    val: slice
    threshold: slice
    test: slice

def temporal_slices(timestamps: pd.Series,
                    train_size: float = 0.70,
                    val_size: float = 0.10,
                    threshold_size: float = 0.05,
                    ) -> SplitBoundaries:
    timestamps = pd.to_datetime(pd.Series(timestamps), errors="raise").reset_index(drop=True)
    if timestamps.empty or timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Expected nonempty, sorted timestamps without NaT")
    fractions = np.asarray([train_size, val_size, threshold_size], dtype=float)
    if not np.isfinite(fractions).all() or (fractions <= 0).any() or fractions.sum() >= 1:
        raise ValueError("Split fractions must be positive and sum to less than 1")
    
    n = len(timestamps)

    train_end, val_end, threshold_end = [
        int(timestamps.searchsorted(timestamps.iloc[min(int(n * fraction), n - 1)], side="left"))
        for fraction in np.cumsum(fractions)
    ]

    train = slice(0, train_end)
    val = slice(train_end, val_end)
    threshold = slice(val_end, threshold_end)
    test = slice(threshold_end, n)

    boundaries = SplitBoundaries(
        train=train,
        val=val,
        threshold=threshold,
        test=test,
    )
    validate_split_boundaries(timestamps, boundaries)
    return boundaries
    


def validate_split_boundaries(timestamps: pd.Series, boundaries: SplitBoundaries) -> dict:
    timestamps = pd.to_datetime(pd.Series(timestamps), errors="raise").reset_index(drop=True)
    splits = {
        "train": boundaries.train,
        "val": boundaries.val,
        "threshold": boundaries.threshold,
        "test": boundaries.test,
    }

    if timestamps.empty:
        raise ValueError("timestamps is empty")
    if timestamps.isna().any():
        raise ValueError("timestamps contains NaN/NaT")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted chronologically with the data")

    previous_stop = 0
    for name, partition in splits.items():
        if not isinstance(partition, slice) or partition.step not in (None, 1):
            raise ValueError(f"{name} must be a contiguous positional slice")
        start, stop = partition.start, partition.stop
        if not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer)):
            raise ValueError(f"{name} must have integer start and stop")
        if start != previous_stop or not start < stop <= len(timestamps):
            raise ValueError(f"{name}: empty, overlapping, missing or out-of-range rows")
        previous_stop = stop
    if previous_stop != len(timestamps):
        raise ValueError("Splits do not cover all observations")
    splits = {name: timestamps.iloc[partition] for name, partition in splits.items()}

    for name, split in splits.items():
        if split.empty:
            raise ValueError(f"{name} is empty")
        if split.isna().any():
            raise ValueError(f"{name} contains NaN/NaT")
        if not split.is_monotonic_increasing:
            raise ValueError(f"{name} is not sorted chronologically")

    # Time border
    names = list(splits)

    for left_name, right_name in zip(names, names[1:]):
        left = splits[left_name]
        right = splits[right_name]

        if left.max() >= right.min():
            raise ValueError(f"Temporal overlap: {left_name} and {right_name}")

    # Absence of index overlaps
    all_indices = pd.Index([], dtype=timestamps.index.dtype)

    for name, split in splits.items():
        if not split.index.is_unique:
            raise ValueError(f"{name} contains duplicate indices")

        if len(all_indices.intersection(split.index)) > 0:
            raise ValueError(f"Index overlap in {name}")

        all_indices = all_indices.append(split.index)

    # Full coverage check
    if not timestamps.index.is_unique:
        raise ValueError("timestamps index must be unique")

    if len(all_indices) != len(timestamps):
        raise ValueError("Splits do not cover all observations")

    if not all_indices.difference(timestamps.index).empty:
        raise ValueError("Splits contain unknown indices")

    if not timestamps.index.difference(all_indices).empty:
        raise ValueError("Some observations are missing")

    # Verification of date consistency with source data
    for name, split in splits.items():
        if not split.equals(timestamps.loc[split.index]):
            raise ValueError(f"{name} contains timestamps inconsistent with source")
    return {
        name: {
            "rows": len(split),
            "min_timestamp": split.iloc[0].isoformat(),
            "max_timestamp": split.iloc[-1].isoformat(),
        }
        for name, split in splits.items()
    }