from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import hashlib

from dataclasses import dataclass

from .schema import RAW_COLUMNS, CATEGORICAL

@dataclass
class ExtractData:
    df: pd.DataFrame
    path: Path
    format: str
    sha256: str
    size_bytes: int
    extracted_rows: int

def extract_dataset(data_path, smoke = False):
    path = Path(data_path)
    if path.suffix == ".parquet":
        if smoke:
            chunks = []
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=100000, columns=RAW_COLUMNS):
                frame = batch.to_pandas()
                chunks.append(frame.iloc[(-offset) % 50::50])
                offset += len(frame)
            df = pd.concat(chunks, ignore_index=True) if chunks else pd.read_parquet(path, columns=RAW_COLUMNS)
        else:
            df = pd.read_parquet(path, columns=RAW_COLUMNS)

    elif path.suffix == ".csv":
        chunks = []

        for chunk in pd.read_csv(
            path, usecols=RAW_COLUMNS, chunksize=100000,
            dtype={column: "string" for column in CATEGORICAL},
        ):
            if smoke:
                chunk = chunk.iloc[::50]

            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)

    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    
    return ExtractData(
        df=df,
        path=path,
        format=path.suffix,
        sha256=calculate_sha256(path),
        size_bytes=path.stat().st_size,
        extracted_rows=len(df),
    )


def calculate_sha256(path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    path = Path(path)
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()