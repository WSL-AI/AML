"""Download only LI-Small_Trans.csv using the official Kaggle CLI"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile


DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
FILENAME = "LI-Small_Trans.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "raw"
REQUIRED_COLUMNS = {
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
}


def validate_csv(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        header = next(reader, [])
        columns = set(header)
        if header.count("Account") == 2:
            columns.add("Account.1")
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Invalid CSV: missing columns {sorted(missing)}")
        row = next(reader, [])
        if not row or len(row) != len(header):
            raise ValueError("CSV has no valid transaction rows")


def download_dataset(output_dir: Path = DEFAULT_OUTPUT) -> Path:
    output_dir = Path(output_dir).resolve()
    target = output_dir / FILENAME
    if target.exists():
        validate_csv(target)
        print(f"File already exists; download skipped: {target}")
        return target

    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise RuntimeError("Kaggle CLI is missing. Install it with: uv tool install kaggle")

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".aml-download-", dir=output_dir) as directory:
        staging = Path(directory)
        process = subprocess.run([
            kaggle, "datasets", "download", "-d", DATASET,
            "-f", FILENAME, "-p", str(staging),
        ])
        if process.returncode != 0:
            raise RuntimeError(
                "Kaggle download failed. Check authentication (kaggle auth login), "
                "dataset access and network connectivity."
            )

        candidate = staging / FILENAME
        if not candidate.is_file():
            archive = staging / f"{FILENAME}.zip"
            if not archive.is_file():
                raise RuntimeError(f"Kaggle did not produce {FILENAME} or its ZIP archive")
            with ZipFile(archive) as zipped:
                members = [
                    member for member in zipped.infolist()
                    if not member.is_dir() and Path(member.filename).name == FILENAME
                ]
                if len(members) != 1:
                    raise ValueError(f"Archive must contain exactly one {FILENAME}")
                with zipped.open(members[0]) as source, candidate.open("xb") as destination:
                    shutil.copyfileobj(source, destination)

        validate_csv(candidate)
        os.link(candidate, target)

    print(f"Dataset ready: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        download_dataset(args.output_dir)
    except (OSError, ValueError, RuntimeError, BadZipFile, UnicodeError) as error:
        print(f"Download error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
