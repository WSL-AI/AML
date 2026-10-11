import importlib.util
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest


spec = importlib.util.spec_from_file_location(
    "download_dataset", Path(__file__).resolve().parents[1] / "scripts" / "download_dataset.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)

CSV = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
    "2025-01-01 00:00,001,000123,002,000456,10,USD,10,USD,Wire,0\n"
)


@pytest.mark.parametrize("zipped", [False, True])
def test_download_single_file_and_skip_existing(tmp_path, monkeypatch, zipped):
    monkeypatch.setattr(download.shutil, "which", lambda name: "/fake/kaggle")
    calls = []

    def run(command):
        calls.append(command)
        assert command[command.index("-f") + 1] == download.FILENAME
        assert command[command.index("-d") + 1] == download.DATASET
        staging = Path(command[command.index("-p") + 1])
        if zipped:
            with ZipFile(staging / f"{download.FILENAME}.zip", "w") as archive:
                archive.writestr("nested/" + download.FILENAME, CSV)
                archive.writestr("../unwanted.txt", "must not be extracted")
        else:
            (staging / download.FILENAME).write_text(CSV)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(download.subprocess, "run", run)
    target = download.download_dataset(tmp_path)
    assert target.read_text() == CSV
    assert sorted(path.name for path in tmp_path.iterdir()) == [download.FILENAME]
    assert download.download_dataset(tmp_path) == target
    assert len(calls) == 1


def test_failed_download_leaves_no_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda name: "/fake/kaggle")
    monkeypatch.setattr(download.subprocess, "run", lambda command: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="Kaggle download failed"):
        download.download_dataset(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_invalid_payload_is_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda name: "/fake/kaggle")

    def run(command):
        staging = Path(command[command.index("-p") + 1])
        (staging / download.FILENAME).write_text("<html>Sign in</html>")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(download.subprocess, "run", run)
    with pytest.raises(ValueError, match="missing columns"):
        download.download_dataset(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_missing_cli_has_install_instructions(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="uv tool install kaggle"):
        download.download_dataset(tmp_path)
