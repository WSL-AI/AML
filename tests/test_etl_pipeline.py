import json
import subprocess
import sys

import pandas as pd
import pytest

from aml.cli.etl import run_etl
from aml.data.read import load_dataset_splits, resolve_run


def source_file(tmp_path):
    n = 1000
    raw = pd.DataFrame({
        "Timestamp": pd.date_range("2025-01-01", periods=n, freq="min"),
        "From Bank": ["001"] * n, "Account": ["000123"] * n,
        "To Bank": ["002"] * n, "Account.1": ["000456"] * n,
        "Receiving Currency": ["USD"] * n, "Payment Currency": ["USD"] * n,
        "Payment Format": ["Wire"] * n,
        "Amount Paid": [float(i + 1) for i in range(n)],
        "Amount Received": [float(i + 1) for i in range(n)],
        "Is Laundering": [i % 2 for i in range(n)],
    })
    source = tmp_path / "raw.csv"
    raw.to_csv(source, index=False)
    return source


def test_etl_roundtrip_and_independent_runs(tmp_path, monkeypatch):
    monkeypatch.delenv("AML_RUN_DIR", raising=False)
    source = source_file(tmp_path)
    runs = tmp_path / "artifacts" / "runs"
    full = run_etl(source, runs)
    smoke = run_etl(source, runs, smoke=True)
    assert full != smoke
    assert (full / "SUCCESS").is_file()
    assert (smoke / "SUCCESS").is_file()
    assert resolve_run(tmp_path) == smoke.resolve()
    loaded = load_dataset_splits(input_dir=full / "data")
    assert sum(len(X) for X, _ in loaded.values()) == 1000
    assert sum(len(X) for X, _ in load_dataset_splits(input_dir=smoke / "data").values()) == 20
    for name, (X, y) in loaded.items():
        assert len(X) == len(y)
        assert "Timestamp" not in X and "Is Laundering" not in X
        assert X.Account.iloc[0] == "000123"
        assert len(pd.read_parquet(full / "data" / f"{name}_time.parquet")) == len(X)
    manifest = json.loads((full / "data_manifest.json").read_text())
    assert manifest["source"]["extracted_rows"] == 1000
    assert json.loads((smoke / "configuration.json").read_text())["smoke"] is True


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
               "--source", str(source), "--output", str(output), "--smoke"]
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "ETL completed:" in process.stdout
    pointer = json.loads((output / "active.json").read_text())
    assert (output / pointer["path"] / "SUCCESS").is_file()
    process = subprocess.run(
        [sys.executable, "-B", "-m", "aml.cli.etl", "--source", str(tmp_path / "missing.csv"),
         "--output", str(output)], capture_output=True, text=True,
    )
    assert process.returncode != 0
    assert json.loads((output / "active.json").read_text()) == pointer
