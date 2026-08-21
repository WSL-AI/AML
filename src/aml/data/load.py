from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATH = PROJECT_ROOT / "data" / "processed"

TARGET = "Is Laundering"
CSV_COLUMNS = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    TARGET,
]
DTYPES = {
    "From Bank": "string",
    "Account": "string",
    "To Bank": "string",
    "Account.1": "string",
    "Amount Received": "float64",
    "Receiving Currency": "string",
    "Amount Paid": "float64",
    "Payment Currency": "string",
    "Payment Format": "string",
    TARGET: "int8",
}

def load_raw_data(file_name: str = "raw", path: str | Path = PATH, dtypes=DTYPES) -> pd.DataFrame:
    """
    Just loads dataset .parquet without feature engineering and splits
    """
    input_path = Path(path)
    if not input_path.exists():
            raise FileNotFoundError(f"Specified directory does not exist: {input_path}")
    df = pd.read_parquet(path / f"{file_name}.parquet")

    df = (
    df.drop_duplicates()
      .sort_values("Timestamp", kind="mergesort")
      .reset_index(drop=True)
    )

    return df

def load_dataset_splits(sets: list = None,
                        input_dir: str | Path = PATH, 
                        target_col: str = "Is Laundering",
                        ) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """
    Loads splits of the dataset from Parquet files, separated by X and Y.
    
    sets: None - if you need to load all splits without raw (train, val, test)
            ["...", "..."] - if you only need a few of them (["train", "val"])
            "all" - if you need absolutely all splits (raw, train, val, test)
    input_dir: path to the folder containing the files.
    target_col: target ("Is Laundering")
    """
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Specified directory does not exist: {input_path}")

    if sets is None:
        target_splits = {p.stem for p in input_path.glob("*.parquet")} - {"raw"}
    elif sets in ("all", ["all"]):
        target_splits = {p.stem for p in input_path.glob("*.parquet")}
    else:
        target_splits = set(sets) if isinstance(sets, list) else {sets}

    splits = {}
    for split_name in target_splits:
        file_path = input_path / f"{split_name}.parquet"
        
        if not file_path.exists():
            continue 
            
        df = pd.read_parquet(file_path)
        splits[split_name] = (df.drop(columns=[target_col]), df[target_col])

    if not splits:
        raise FileNotFoundError(
            f"В директории {input_path} не найдено .parquet файлов."
        )
    
    return splits