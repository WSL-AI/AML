import pandas as pd

def rolling_windows(timestamp, n_folds=3, initial_days=3, validation_days=1, evaluation_days=1):
    """Build expanding training and fixed-duration validation/evaluation windows."""
    from datetime import timedelta

    t = pd.Series(timestamp).reset_index(drop=True)
    if t.empty or t.isna().any() or not t.is_monotonic_increasing:
        raise ValueError("Expected nonempty sorted timestamps")
    if min(n_folds, initial_days, validation_days, evaluation_days) <= 0:
        raise ValueError("Window sizes must be positive")

    result = []
    for fold in range(n_folds):
        train_end = t.iloc[0].floor("D").to_pydatetime() + timedelta(
            days=initial_days + fold * evaluation_days
        )
        val_end = train_end + timedelta(days=validation_days)
        eval_end = val_end + timedelta(days=evaluation_days)
        if eval_end > t.iloc[-1]:
            raise ValueError("Insufficient development period for complete backtest windows")
        edges = [0] + [int(t.searchsorted(value, side="left")) for value in (train_end, val_end, eval_end)]
        if any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("Empty backtest partition")
        result.append(dict(zip(
            ("train", "val", "evaluation"),
            (slice(a, b) for a, b in zip(edges, edges[1:])),
        )))
    return result
