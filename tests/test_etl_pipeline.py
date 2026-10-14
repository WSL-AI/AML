import json
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest

from aml.cli.etl import run_etl
from aml.data.read import load_dataset_splits, load_run_raw, resolve_run, run_periods
from aml.data.validate import validate_raw
from conftest import make_transactions, write_ibm_csv


def source_file(tmp_path):
    raw = make_transactions(days=12, rows=8_000, accounts=600, seed=1)
    raw.loc[0, "Account"] = "000123"
    return write_ibm_csv(raw, tmp_path / "raw.csv")


def test_etl_roundtrip_and_independent_runs(tmp_path, monkeypatch):
    monkeypatch.delenv("AML_RUN_DIR", raising=False)
    source = source_file(tmp_path)
    runs = tmp_path / "artifacts" / "runs"
    full = run_etl(source, runs)
    smoke = run_etl(source, runs, smoke=True)
    assert full != smoke
    assert (full / "SUCCESS").is_file() and (smoke / "SUCCESS").is_file()
    assert resolve_run(tmp_path) == smoke.resolve()

    manifest = json.loads((full / "data_manifest.json").read_text())
    excluded = sum(manifest["transformation"]["excluded_rows"].values())
    loaded = load_dataset_splits(input_dir=full / "data")
    assert sum(len(X) for X, _ in loaded.values()) + excluded == 8_000
    assert manifest["transformation"]["excluded_rows"]["warmup"] > 0
    for name, (X, y) in loaded.items():
        assert len(X) == len(y) == len(pd.read_parquet(full / "data" / f"{name}_time.parquet"))
        assert "Timestamp" not in X and "Is Laundering" not in X
        assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)
        assert np.isfinite(X.to_numpy()).all()
        assert X.columns.tolist() == manifest["features"]
    assert (full / "encoder.joblib").is_file()
    assert manifest["transformation"]["encoding"]["fit_period"] == "train"
    assert json.loads((smoke / "configuration.json").read_text())["smoke"] is True
    smoke_rows = sum(len(X) for X, _ in load_dataset_splits(input_dir=smoke / "data").values())
    assert 0 < smoke_rows < 8_000 / 10


def test_run_periods_reproduce_saved_splits(tmp_path):
    run = run_etl(source_file(tmp_path), tmp_path / "runs")
    raw = load_run_raw(run)
    periods = run_periods(run, raw["Timestamp"])
    for name in ("train", "val", "threshold", "test"):
        saved = pd.read_parquet(run / "data" / f"{name}_time.parquet")["Timestamp"]
        pd.testing.assert_series_equal(raw["Timestamp"].iloc[periods[name]].reset_index(drop=True), saved,
                                       check_dtype=False)


def test_saved_pipeline_scores_later_batches(tmp_path):
    raw = make_transactions(days=14, rows=9_000, accounts=600, seed=2)
    history, later = raw[raw["Timestamp"] < "2025-01-13"], raw[raw["Timestamp"] >= "2025-01-13"]
    run = run_etl(write_ibm_csv(history, tmp_path / "raw.csv"), tmp_path / "runs")
    pipeline = joblib.load(run / "feature_pipeline.joblib")
    encoder = joblib.load(run / "encoder.joblib")
    scored = encoder.transform(pipeline.transform(validate_raw(later).df.drop(columns="Is Laundering")))
    X_test, _ = load_dataset_splits("test", input_dir=run / "data")["test"]
    assert scored.columns.tolist() == X_test.columns.tolist()
    assert len(scored) == len(later)


def test_failed_write_preserves_active_run(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    runs = tmp_path / "runs"
    completed = run_etl(source, runs)
    previous_pointer = (runs / "active.json").read_bytes()

    def fail(*args, **kwargs):
        raise OSError("Simulated write failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail)
    with pytest.raises(OSError, match="Simulated"):
        run_etl(source, runs)
    assert (runs / "active.json").read_bytes() == previous_pointer
    incomplete = [path for path in runs.iterdir() if path.is_dir() and path != completed]
    assert len(incomplete) == 1
    assert not (incomplete[0] / "SUCCESS").exists()


def test_cli_module_runs_and_fails_on_missing_source(tmp_path):
    source = source_file(tmp_path)
    output = tmp_path / "runs"
    command = [sys.executable, "-B", "-m", "aml.cli.etl",
               "--source", str(source), "--output", str(output), "--smoke", "--warmup-days", "2"]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "ETL completed:" in process.stdout
    pointer = json.loads((output / "active.json").read_text())
    configuration = json.loads((output / pointer["path"] / "configuration.json").read_text())
    assert configuration["split"]["warmup_units"] == 2
    process = subprocess.run(
        [sys.executable, "-B", "-m", "aml.cli.etl", "--source", str(tmp_path / "missing.csv"),
         "--output", str(output)], capture_output=True, text=True,
    )
    assert process.returncode != 0
    assert json.loads((output / "active.json").read_text()) == pointer


@pytest.mark.parametrize("module", ["aml.features", "aml.data", "aml.data.load", "aml.features.pipeline"])
def test_packages_import_in_any_order(module):
    # joblib imports aml.features first when unpickling a saved pipeline.
    process = subprocess.run([sys.executable, "-c", f"import {module}; import aml.data; aml.data.EtlConfig"],
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


def test_runs_from_previous_etl_version_fail_clearly(tmp_path):
    run = run_etl(source_file(tmp_path), tmp_path / "runs")
    manifest = json.loads((run / "data_manifest.json").read_text())
    del manifest["transformation"]["period_starts"]
    (run / "data_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="predates calendar periods"):
        load_run_raw(run)
    with pytest.raises(ValueError, match="predates calendar periods"):
        run_periods(run, pd.Series(pd.to_datetime(["2025-01-01"])))
