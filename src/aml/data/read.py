from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from .schema import CATEGORICAL, TARGET, TIMESTAMP
from .split import boundaries_from_starts
from .validate import validate_raw

PROJECT_ROOT = Path(__file__).resolve().parents[3]

__all__ = [
    "CATEGORICAL", "PROJECT_ROOT", "TARGET",
    "load_dataset_splits", "load_run_raw", "load_split", "load_timestamps",
    "resolve_etl_run", "resolve_run", "run_periods",
]


def resolve_run(project_root: str | Path = PROJECT_ROOT) -> Path:
    """Selected run: ``AML_RUN_DIR`` if set, otherwise ``artifacts/runs/active.json``."""
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


def resolve_etl_run(project_root=PROJECT_ROOT) -> Path:
    """Find the source ETL run even when the active run contains encoded features."""
    return _etl_run(resolve_run(project_root))


def load_run_raw(run_dir) -> pd.DataFrame:
    """Re-read the exact source and sampling mode recorded by the run's ETL."""
    from .extract import extract_dataset

    run = _current_etl_run(Path(run_dir))
    source = _manifest(run).get("source")
    if not isinstance(source, dict) or "path" not in source:
        raise ValueError("Source provenance missing: run notebooks/00_etl.ipynb first")
    config = json.loads((run / "configuration.json").read_text())
    extracted = extract_dataset(source["path"], smoke=config["smoke"])
    if extracted.sha256 != source["sha256"] or extracted.extracted_rows != source["extracted_rows"]:
        raise ValueError("Raw source changed since ETL; rerun 00_etl.ipynb")
    return validate_raw(extracted.df).df


def run_periods(run_dir, timestamps: pd.Series) -> dict[str, slice]:
    """Positional slices of warmup/train/val/threshold/test/tail as cut by the run's ETL."""
    starts = _manifest(_current_etl_run(Path(run_dir)))["transformation"]["period_starts"]
    return boundaries_from_starts(timestamps, starts).slices()


def load_dataset_splits(sets=None, input_dir=None, target_col=TARGET):
    """Load explicit splits deterministically; missing files are errors, not omissions.

    None loads train/val/threshold/test; 'all' discovers labeled parquet files,
    excluding the timestamp sidecars.
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
        raise FileNotFoundError(f"No dataset splits in {path}")
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


def load_split(name, data_dir, target_col=TARGET):
    """Load one named feature split and its target."""
    return load_dataset_splits(name, input_dir=data_dir, target_col=target_col)[name]


def load_timestamps(name, data_dir) -> pd.Series:
    return pd.read_parquet(Path(data_dir) / f"{name}_time.parquet")[TIMESTAMP]


def _etl_run(run: Path) -> Path:
    manifest = _manifest(run)
    if "etl_run_dir" in manifest:
        run = Path(manifest["etl_run_dir"])
    if not (run / "SUCCESS").is_file():
        raise ValueError("Run notebooks/00_etl.ipynb first: no completed ETL run selected")
    return run.resolve()


def _current_etl_run(run: Path) -> Path:
    """ETL run written by the calendar-period pipeline; older runs use other smoke sampling and splits."""
    run = _etl_run(run)
    if "period_starts" not in _manifest(run).get("transformation", {}):
        raise ValueError(f"{run.name} predates calendar periods and sender-based smoke sampling; rerun ETL")
    return run


def _manifest(run: Path) -> dict:
    return json.loads((Path(run) / "data_manifest.json").read_text(encoding="utf-8"))
