from __future__ import annotations

import pandas as pd
from pathlib import Path
import hashlib

from dataclasses import dataclass

from .schema import RAW_COLUMNS

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
        df = pd.read_parquet(path, columns=RAW_COLUMNS)
        if smoke:
            df = df.iloc[::50].reset_index(drop=True)

    elif path.suffix == ".csv":
        chunks = []

        for chunk in pd.read_csv(path, usecols=RAW_COLUMNS, chunksize=100_000,):
            if smoke:
                chunk = chunk.iloc[::50]

            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)

    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    if smoke:
        return ExtractData(df=df)
    
    return ExtractData(
        df=df,
        path=path,
        format=path.suffix,
        sha256=calculate_sha256(path),
        size_bytes=path.stat().st_size,
        extracted_rows=len(df),
    )


def calculate_sha256(path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()