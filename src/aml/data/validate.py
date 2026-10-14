from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import (
    AMOUNT, CATEGORICAL, PAYMENT_FORMATS, RAW_COLUMNS, RECEIVER, SENDER, TARGET, TIMESTAMP,
    TIMESTAMP_FORMAT,
)


@dataclass(frozen=True)
class ValidationResult:
    df: pd.DataFrame
    quality_report: dict


def validate_raw(df: pd.DataFrame) -> ValidationResult:
    """Enforce the raw contract and return chronologically sorted transactions.

    Identical rows are kept: without a transaction ID they may be genuine repeats.
    """
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")

    result = df.loc[:, RAW_COLUMNS].copy()
    if result.empty:
        raise ValueError("Raw data is empty")
    null_columns = result.columns[result.isna().any()].tolist()
    if null_columns:
        raise ValueError(f"Missing values in: {null_columns}")

    result[TIMESTAMP] = _parse_timestamps(result[TIMESTAMP])
    for column in CATEGORICAL:
        result[column] = result[column].astype("string")
        if result[column].eq("").any():
            raise ValueError(f"Empty identifiers in: {column}")
    for column in AMOUNT:
        result[column] = pd.to_numeric(result[column], errors="raise").astype("float64")
        if not np.isfinite(result[column]).all() or (result[column] < 0).any():
            raise ValueError(f"Invalid amount: {column}")
    if not result[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be binary")
    result[TARGET] = result[TARGET].astype("int8")

    unknown_formats = set(result["Payment Format"].unique()) - PAYMENT_FORMATS
    if unknown_formats:
        raise ValueError(f"Unknown payment formats: {sorted(unknown_formats)}")

    result = result.sort_values(TIMESTAMP, kind="stable").reset_index(drop=True)
    return ValidationResult(df=result, quality_report=create_quality_report(result))


def _parse_timestamps(values: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(values):
        return values.astype("datetime64[ns]")
    try:
        return pd.to_datetime(values, format=TIMESTAMP_FORMAT)
    except (TypeError, ValueError):
        return pd.to_datetime(values, format="ISO8601")


def create_quality_report(df: pd.DataFrame) -> dict:
    daily = df.groupby(df[TIMESTAMP].dt.floor("D"))[TARGET].agg(["size", "sum"])
    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "zero_amounts": int((df[AMOUNT] == 0).any(axis=1).sum()),
        "self_transfers": int(
            (df[SENDER[0]].eq(df[RECEIVER[0]]) & df[SENDER[1]].eq(df[RECEIVER[1]])).sum()
        ),
        "negatives": int((df[TARGET] == 0).sum()),
        "positives": int((df[TARGET] == 1).sum()),
        "positive_rate": float(df[TARGET].mean()),
        "min_timestamp": df[TIMESTAMP].min().isoformat(),
        "max_timestamp": df[TIMESTAMP].max().isoformat(),
        "payment_formats": _counts(df["Payment Format"]),
        "currencies": _counts(pd.concat([df["Payment Currency"], df["Receiving Currency"]])),
        "daily": [
            {"day": day.date().isoformat(), "rows": int(row["size"]), "positives": int(row["sum"])}
            for day, row in daily.iterrows()
        ],
    }


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(key): int(count) for key, count in values.value_counts().sort_index().items()}
