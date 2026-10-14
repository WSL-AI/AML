from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq

from .keys import account_key
from .schema import AMOUNT, CATEGORICAL, RAW_COLUMNS, SENDER, TARGET, TIMESTAMP, TIMESTAMP_FORMAT

# Smoke keeps complete histories of every 50th sender instead of every 50th row:
# row sampling would break the account histories the features depend on.
SMOKE_SENDER_RATE = 50


@dataclass(frozen=True)
class ExtractData:
    df: pd.DataFrame
    path: Path
    format: str
    sha256: str
    size_bytes: int
    extracted_rows: int


def extract_dataset(data_path, smoke: bool = False) -> ExtractData:
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    sha256 = calculate_sha256(path)

    if path.suffix == ".parquet":
        df = pq.read_table(path, columns=RAW_COLUMNS).to_pandas()
    elif path.suffix == ".csv":
        df = _read_csv(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    if smoke:
        df = sample_senders(df)

    return ExtractData(
        df=df,
        path=path,
        format=path.suffix,
        sha256=sha256,
        size_bytes=path.stat().st_size,
        extracted_rows=len(df),
    )


def sample_senders(df: pd.DataFrame, rate: int = SMOKE_SENDER_RATE) -> pd.DataFrame:
    """Deterministically keep all transactions of a 1/rate subset of sender accounts."""
    keys = account_key(df[SENDER[0]], df[SENDER[1]])
    return df.loc[keys % rate == 0].reset_index(drop=True)


def _read_csv(path: Path) -> pd.DataFrame:
    names = _csv_column_names(path)
    column_types = {
        **{column: pa.string() for column in CATEGORICAL},
        **{column: pa.float64() for column in AMOUNT},
        TIMESTAMP: pa.timestamp("s"),
        TARGET: pa.int8(),
    }
    table = pv.read_csv(
        path,
        read_options=pv.ReadOptions(column_names=names, skip_rows=1),
        convert_options=pv.ConvertOptions(
            column_types=column_types,
            include_columns=RAW_COLUMNS,
            timestamp_parsers=[TIMESTAMP_FORMAT, pv.ISO8601],
            strings_can_be_null=False,
        ),
    )
    return table.to_pandas(types_mapper={pa.string(): pd.StringDtype()}.get)


def _csv_column_names(path: Path) -> list[str]:
    """Read the header and give the receiver account its pandas name, Account.1."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file), [])
    names = []
    for name in header:
        names.append("Account.1" if name == "Account" and "Account" in names else name)
    if len(set(names)) != len(names) or set(names) != set(RAW_COLUMNS):
        raise ValueError(f"Unexpected CSV header: {header}")
    return names


def calculate_sha256(path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    hasher = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()
