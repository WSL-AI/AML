"""Notebook data cells executed against a small synthetic ETL run."""
import json
from pathlib import Path

import pytest

from aml.cli.etl import run_etl
from aml.data.read import load_run_raw, resolve_etl_run, run_periods
from conftest import make_transactions, write_ibm_csv

ROOT = Path(__file__).resolve().parents[1]


def notebook(name):
    return json.loads((ROOT / "notebooks" / f"{name}.ipynb").read_text())


def tagged(nb, tag):
    """Source of the code cell carrying ``tag``; tags survive cells being added or moved."""
    cells = [cell for cell in nb["cells"] if tag in cell.get("metadata", {}).get("tags", [])]
    assert len(cells) == 1, f"expected one cell tagged {tag!r}"
    return "".join(cells[0]["source"])


def source_file(tmp_path):
    raw = make_transactions(days=12, rows=6_000, accounts=300, seed=3)
    raw.loc[0, "Account"] = "0001"
    return write_ibm_csv(raw, tmp_path / "HI-Small_Trans.csv")


def test_eda_runs_on_history_before_validation(tmp_path, monkeypatch):
    source = source_file(tmp_path)
    etl = run_etl(source, tmp_path / "artifacts" / "runs")
    encoded = tmp_path / "artifacts" / "runs" / "encoded"
    encoded.mkdir()
    (encoded / "data_manifest.json").write_text(json.dumps({"etl_run_dir": str(etl)}))
    monkeypatch.setenv("AML_RUN_DIR", str(encoded))
    assert resolve_etl_run(tmp_path) == etl
    loaded = load_run_raw(encoded)
    assert len(loaded) == 6_000 and "0001" in set(loaded["Account"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    ns = {"display": lambda *args: None}
    for cell in notebook("01_eda")["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), ns)
    validation_start = run_periods(etl, loaded["Timestamp"])["val"].start
    assert len(ns["df"]) == validation_start
    assert ns["df"]["Timestamp"].max() < loaded["Timestamp"].iloc[validation_start]

    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="changed since ETL"):
        load_run_raw(encoded)


def test_notebook_etl_encoding_and_training_handoff(tmp_path, monkeypatch):
    ns = {"display": lambda *args: None}
    etl_notebook = notebook("00_etl")
    exec(tagged(etl_notebook, "setup"), ns)
    ns.update(SOURCE=source_file(tmp_path), OUTPUT_DIR=tmp_path / "artifacts/runs", SMOKE=False)
    exec(tagged(etl_notebook, "etl-run"), ns)
    exec(tagged(etl_notebook, "etl-check"), ns)
    etl = ns["RUN_DIR"]

    monkeypatch.setenv("AML_RUN_DIR", str(etl))
    encoding = notebook("02_feature_engineering")
    artifacts = f"Path({str(tmp_path)!r}) / \"artifacts\""
    exec(tagged(encoding, "setup").replace('PROJECT_ROOT / "artifacts"', artifacts), ns)
    exec(tagged(encoding, "load"), ns)
    ns["categorical_features"] = ns["X_train"].select_dtypes(include="string").columns.tolist()
    ns["target_encoding_interactions"] = [("From Bank", "Account")]
    exec(tagged(encoding, "export").replace('PROJECT_ROOT / "artifacts/runs"',
                                            f'Path({str(tmp_path)!r}) / "artifacts/runs"'), ns)
    encoded = ns["RUN_DIR"]
    assert encoded != etl
    monkeypatch.setenv("AML_RUN_DIR", str(encoded))
    assert resolve_etl_run(ROOT) == etl
    assert len(load_run_raw(encoded)) == 6_000

    training = notebook("03_training")
    exec(tagged(training, "setup"), ns)
    exec(tagged(training, "load"), ns)
    exec(tagged(training, "sampling"), ns)
    manifest = json.loads((etl / "data_manifest.json").read_text())["transformation"]["periods"]
    assert ns["X_train"].shape[0] == manifest["train"]["rows"]
    assert ns["X_val"].shape[0] == manifest["val"]["rows"]
    assert ns["y_fit"].sum() == ns["y_train"].sum()


def test_training_accepts_etl_without_encoding_notebook(tmp_path, monkeypatch):
    etl = run_etl(source_file(tmp_path), tmp_path / "runs")
    monkeypatch.setenv("AML_RUN_DIR", str(etl))
    ns = {}
    training = notebook("03_training")
    for tag in ("setup", "load", "sampling"):
        exec(tagged(training, tag), ns)
    assert ns["RUN_DIR"] == etl
    assert ns["manifest"]["run_id"] == etl.name
    assert ns["X_train"].columns.tolist() == ns["manifest"]["features"]
