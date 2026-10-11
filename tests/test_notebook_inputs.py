import json
from pathlib import Path

import pandas as pd
import pytest

from aml.cli.etl import run_etl
from aml.data.read import load_run_raw, resolve_etl_run


def test_notebook_source_lineage_and_changed_source(tmp_path, monkeypatch):
    raw = pd.DataFrame({
        "Timestamp": pd.date_range("2025-01-01", periods=100, freq="min"),
        "From Bank": ["001"] * 100, "Account": ["0001"] * 100,
        "To Bank": ["002"] * 100, "Account.1": ["0002"] * 100,
        "Receiving Currency": ["USD"] * 100, "Payment Currency": ["USD"] * 100,
        "Payment Format": ["Wire"] * 100,
        "Amount Paid": [1.0] * 100, "Amount Received": [1.0] * 100,
        "Is Laundering": [0, 1] * 50,
    })
    source = tmp_path / "HI-Small_Trans.csv"
    raw.to_csv(source, index=False)
    runs = tmp_path / "artifacts" / "runs"
    etl = run_etl(source, runs, smoke=False)
    encoded = runs / "encoded"
    encoded.mkdir()
    (encoded / "data_manifest.json").write_text(json.dumps({"etl_run_dir": str(etl)}))
    monkeypatch.setenv("AML_RUN_DIR", str(encoded))
    assert resolve_etl_run(tmp_path) == etl
    loaded = load_run_raw(encoded)
    assert len(loaded) == 100
    assert loaded.Account.iloc[0] == "0001"
    # EDA must run top-to-bottom in a fresh namespace, without hidden df/imports.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    notebook = Path(__file__).resolve().parents[1] / "notebooks/01_eda.ipynb"
    ns = {"display": lambda *args: None}
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), ns)
    assert len(ns["df"]) == 70
    assert ns["df"].Timestamp.max() < loaded.Timestamp.iloc[70]
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="changed since ETL"):
        load_run_raw(encoded)


def test_notebook_etl_and_encoding_handoff(tmp_path, monkeypatch):
    """Execute the actual notebook data cells on a tiny complete dataset."""
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "HI-Small_Trans.csv"
    raw = pd.DataFrame({
        "Timestamp": pd.date_range("2025-01-01", periods=200, freq="min"),
        "From Bank": ["001"] * 200, "Account": ["0001"] * 200,
        "To Bank": ["002"] * 200, "Account.1": ["0002"] * 200,
        "Receiving Currency": ["USD"] * 200, "Payment Currency": ["USD"] * 200,
        "Payment Format": ["Wire"] * 200,
        "Amount Paid": [1.0] * 200, "Amount Received": [1.0] * 200,
        "Is Laundering": [0, 1] * 100,
    })
    raw.to_csv(source, index=False)
    ns = {"display": lambda *args: None}
    nb = json.loads((root / "notebooks/00_etl.ipynb").read_text())
    exec("".join(nb["cells"][1]["source"]), ns)
    ns.update(SOURCE=source, OUTPUT_DIR=tmp_path / "artifacts/runs", SMOKE=False)
    exec("".join(nb["cells"][5]["source"]), ns)
    exec("".join(nb["cells"][7]["source"]), ns)
    etl = ns["RUN_DIR"]
    monkeypatch.setenv("AML_RUN_DIR", str(etl))
    nb = json.loads((root / "notebooks/02_feature_engineering.ipynb").read_text())
    setup = "".join(nb["cells"][2]["source"])
    setup = setup.replace('new_run(PROJECT_ROOT / "artifacts", "runs")',
                          f'new_run(Path({str(tmp_path)!r}) / "artifacts", "runs")')
    exec(setup, ns)
    exec("".join(nb["cells"][3]["source"]), ns)
    ns["categorical_features"] = ns["X_train"].select_dtypes(include="string").columns.tolist()
    ns["target_encoding_interactions"] = []
    export = "".join(nb["cells"][15]["source"])
    export = export.replace('PROJECT_ROOT / "artifacts/runs"', f'Path({str(tmp_path)!r}) / "artifacts/runs"')
    exec(export, ns)
    encoded = ns["RUN_DIR"]
    assert encoded != etl
    monkeypatch.setenv("AML_RUN_DIR", str(encoded))
    assert resolve_etl_run(root) == etl
    assert len(load_run_raw(encoded)) == 200
    nb = json.loads((root / "notebooks/03_training.ipynb").read_text())
    exec("".join(nb["cells"][3]["source"]), ns)
    exec("".join(nb["cells"][5]["source"]), ns)
    assert ns["X_train"].shape[0] == 140
    assert ns["X_val"].shape[0] == 20
