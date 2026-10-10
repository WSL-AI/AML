from __future__ import annotations

import pandas as pd 
import numpy as np

from dataclasses import dataclass

from .schema import RAW_COLUMNS, TIMESTAMP, TARGET, CATEGORICAL

@dataclass
class ValidationResult:
    df: pd.DataFrame
    quality_report: dict

def validate_raw(df):
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {missing}")

    result = df.loc[:, RAW_COLUMNS].copy()
    result[TIMESTAMP] = pd.to_datetime(result["Timestamp"], errors="raise")

    if result.empty or result.isna().any().any():
        raise ValueError("Raw data must be nonempty and contain no missing values")
    if not result[TARGET].isin([0, 1]).all():
        raise ValueError("Target must be binary")

    for column in ("Amount Paid", "Amount Received"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column]).all() or (result[column] < 0).any():
            raise ValueError(f"Invalid amount: {column}")

    for column in CATEGORICAL:
        result[column] = result[column].astype("string")
    result[TARGET] = result[TARGET].astype("int8")

    result = result.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
    quality_report=create_quality_report(result)

    return ValidationResult(df=result, quality_report=quality_report)


def create_quality_report(df: pd.DataFrame) -> dict:
    quality_report = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": {column: int(count) for column, count in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "negative_amounts": int((df[["Amount Paid", "Amount Received"]] < 0).any(axis=1).sum()),
        "negatives": int((df[TARGET] == 0).sum()),
        "positives": int((df[TARGET] == 1).sum()),
        "positive_rate": float(df[TARGET].mean()),
        "min_timestamp": df[TIMESTAMP].min().isoformat(),
        "max_timestamp": df[TIMESTAMP].max().isoformat(),
    }
    return quality_report