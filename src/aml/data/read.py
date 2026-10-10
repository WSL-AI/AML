from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import TARGET, CATEGORICAL, RAW_COLUMNS
from .validate import validate_raw as _validate_raw
from .split import temporal_slices as _temporal_slices

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATH = PROJECT_ROOT / "data" / "processed"
SPLIT_PATH = PROJECT_ROOT / "artifacts" / "runs" / "full" / "data"

CSV_COLUMNS = RAW_COLUMNS

def load_dataset_splits(sets=None, input_dir=None, target_col=TARGET):
    """Load explicit splits deterministically; missing files are errors, not omissions.

    None loads train/val/threshold/test. 'all' discovers labeled parquet files,
    excluding the timestamp sidecars. Raw data is included only if present and
    explicitly requested with 'all' or 'raw'.
    """
    path = Path(input_dir) if input_dir is not None else resolve_run(PROJECT_ROOT) / "data"
    if not path.is_dir():
        raise FileNotFoundError(path)
    if sets is None:
        names = ["train", "val", "threshold", "test"]
    elif sets == "all" or sets == ["all"]:
        names = sorted(p.stem for p in path.glob("*.parquet") if not p.stem.endswith("_time"))
    else:
        names = [sets] if isinstance(sets, str) else list(sets)
    if not names:
        raise FileNotFoundError(f'No dataset splits in {path}')
    result = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError("Expected a split name, not a path")
        file_path = path / f"{name}.parquet"
        if not file_path.is_file():
            raise FileNotFoundError(file_path)
        frame = pd.read_parquet(file_path)
        if target_col not in frame:
            raise ValueError(f"{name}: missing target {target_col}")
        result[name] = (frame.drop(columns=[target_col]), frame[target_col])
    return result

def load_raw_data(file_name="raw", path=PATH, dtypes=None):
    """Read validated raw Parquet; retain identical rows without a transaction ID."""
    df = pd.read_parquet(Path(path) / f'{file_name}.parquet', columns=RAW_COLUMNS)
    if dtypes is not None:
        df = df.astype(dtypes)
    return _validate_raw(df).df


def load_raw(root=PROJECT_ROOT, smoke=False):
    """Load raw transactions from Parquet, or fall back to the source CSV"""
    import pyarrow.parquet as pq

    root = Path(root)
    path = root / "data" / "processed" / "raw.parquet"
    if path.exists():
        if smoke:
            chunks = []
            offset = 0
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=100_000,
                columns=RAW_COLUMNS,
            ):
                frame = batch.to_pandas()
                positions = np.flatnonzero((np.arange(len(frame)) + offset) % 50 == 0)
                chunks.append(frame.iloc[positions])
                offset += len(frame)
            df = pd.concat(chunks, ignore_index=True)
        else:
            df = pd.read_parquet(path, columns=RAW_COLUMNS)
    else:
        path = root / "data" / "raw" / "LI-Small_Trans.csv"
        chunks = pd.read_csv(
            path,
            usecols=RAW_COLUMNS,
            dtype={column: "string" for column in CATEGORICAL},
            chunksize=100_000,
        )
        frames = []
        offset = 0
        for chunk in chunks:
            chunk_size = len(chunk)
            if smoke:
                positions = np.flatnonzero((np.arange(chunk_size) + offset) % 50 == 0)
                chunk = chunk.iloc[positions]
            frames.append(chunk)
            offset += chunk_size
        df = pd.concat(frames, ignore_index=True)
    return _validate_raw(df).df

def resolve_run(project_root: str | Path = PROJECT_ROOT) -> Path:
    root = Path(project_root)
    configured = os.environ.get("AML_RUN_DIR")
    if configured:
        run_dir = Path(configured).expanduser()
        if not run_dir.is_absolute():
            run_dir = root / run_dir
    else:
        runs_dir = root / "artifacts" / "runs"
        active_path = runs_dir / "active.json"
        if active_path.is_file():
            active = json.loads(active_path.read_text(encoding="utf-8"))
            selected = active if isinstance(active, str) else active.get("path", active.get("run_dir"))
            if not selected:
                raise ValueError(f"Invalid run pointer: {active_path}")
            run_dir = Path(selected)
            if not run_dir.is_absolute():
                relative_to_runs = runs_dir / run_dir
                run_dir = relative_to_runs if relative_to_runs.exists() else root / run_dir
        else:
            run_dir = runs_dir / "full"

    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    return run_dir.resolve()


def temporal_slices(timestamp):
    """Compatibility API used by notebooks: mapping of positional slices."""
    from dataclasses import asdict
    return asdict(_temporal_slices(timestamp))


def load_split(name, data_dir, target_col=TARGET):
    """Load one named feature split and its target."""
    return load_dataset_splits(name, input_dir=data_dir, target_col=target_col)[name]


def scale_train_set(X_train, y_train, negative_sample_ratio=0.39):
    if not 0 < negative_sample_ratio <= 1:
        raise ValueError("negative_sample_ratio must be in (0, 1]")
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows")

    y_array = y_train.to_numpy()
    pos_mask = np.flatnonzero(y_array == 1)
    neg_mask = np.flatnonzero(y_array == 0)
    if not len(pos_mask) or not len(neg_mask):
        raise ValueError("Training data must contain both target classes")

    rng = np.random.default_rng(42)
    neg_size = max(1, int(negative_sample_ratio * len(neg_mask)))
    sampled_neg_id = rng.choice(neg_mask, size=neg_size, replace=False)

    fit_positions = np.sort(
        np.concatenate([pos_mask, sampled_neg_id])
    )
    X_fit = X_train.iloc[fit_positions].reset_index(drop=True)
    y_fit = y_train.iloc[fit_positions].reset_index(drop=True)

    negative_weight = len(neg_mask) / neg_size

    sample_weight = np.where(
        y_fit.to_numpy() == 0,
        negative_weight,
        1.0,
    ).astype("float32")

    return X_fit, y_fit, sample_weight
