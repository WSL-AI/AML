from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

from .transform import TransformResult
from .extract import ExtractData

from .schema import TARGET
from .read import (
    CATEGORICAL, RAW_COLUMNS, CSV_COLUMNS, PROJECT_ROOT, PATH, SPLIT_PATH,
    load_raw_data, load_raw, load_dataset_splits, load_split,
    resolve_run, scale_train_set, temporal_slices, validate_raw,
)


def load_etl_result(
    result: TransformResult,
    output_dir: Path,
    source_metadata: ExtractData,
    configuration: dict | None = None,
) -> Path:
    """Save an immutable run, publishing its pointer only after completion."""
    names = ("train", "val", "threshold", "test")
    if set(result.splits) != set(names):
        raise ValueError("Expected train, val, threshold and test splits")
    previous_end = None
    for name in names:
        split = result.splits[name]
        if not len(split.X) or not len(split.X) == len(split.y) == len(split.timestamps):
            raise ValueError(f"{name}: empty split or inconsistent row counts")
        if not split.X.index.equals(split.y.index) or not split.X.index.equals(split.timestamps.index):
            raise ValueError(f"{name}: inconsistent row indices")
        if TARGET in split.X or "Timestamp" in split.X:
            raise ValueError(f"{name}: target or timestamp included in features")
        if split.X.columns.tolist() != result.feature_names:
            raise ValueError(f"{name}: inconsistent feature columns")
        if {column: str(dtype) for column, dtype in split.X.dtypes.items()} != result.feature_schema:
            raise ValueError(f"{name}: inconsistent feature dtypes")
        if not split.y.isin([0, 1]).all():
            raise ValueError(f"{name}: target must be binary")
        if split.timestamps.isna().any() or not split.timestamps.is_monotonic_increasing:
            raise ValueError(f"{name}: invalid timestamps")
        if previous_end is not None and previous_end >= split.timestamps.iloc[0]:
            raise ValueError(f"{name}: temporal overlap")
        previous_end = split.timestamps.iloc[-1]

    configuration = {} if configuration is None else configuration
    manifest = {
        "source": {
            "path": str(source_metadata.path.resolve()),
            "format": source_metadata.format,
            "sha256": source_metadata.sha256,
            "size_bytes": source_metadata.size_bytes,
            "extracted_rows": source_metadata.extracted_rows,
        },
        "transformation": result.metadata,
    }
    # Reject invalid metadata before allocating an output directory.
    for metadata in (manifest, configuration, result.feature_schema, result.metadata["quality_report"]):
        json.dumps(metadata, allow_nan=False)
    if sum(len(split.X) for split in result.splits.values()) != source_metadata.extracted_rows:
        raise ValueError("Splits do not cover extracted rows")
    parent = Path(output_dir)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12]
    run_dir = parent / run_id
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=False)

    for name, split in result.splits.items():
        frame = split.X.copy()
        frame[TARGET] = split.y.to_numpy()
        frame.to_parquet(data_dir / f"{name}.parquet", engine="pyarrow", index=False)

        split.timestamps.to_frame(name="Timestamp").to_parquet(
            data_dir / f"{name}_time.parquet",
            index=False,
        )
        for filename in (f"{name}.parquet", f"{name}_time.parquet"):
            if pq.ParquetFile(data_dir / filename).metadata.num_rows != len(split.X):
                raise ValueError(f"{filename}: saved row count differs")

    save_json(run_dir / "feature_schema.json", result.feature_schema)
    save_json(run_dir / "quality_report.json", result.metadata["quality_report"])
    save_json(run_dir / "data_manifest.json", manifest)
    save_json(run_dir / "configuration.json", configuration)
    (run_dir / "SUCCESS").write_text("Completed\n", encoding="utf-8")
    temporary = parent / f"active.{uuid.uuid4().hex}.tmp"
    save_json(temporary, {"path": run_id})
    temporary.replace(parent / "active.json")
    return run_dir

def save_json(path: Path, data: dict) -> None:
    content = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    Path(path).write_text(content + "\n", encoding="utf-8")