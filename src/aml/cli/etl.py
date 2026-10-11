from __future__ import annotations

import argparse
from pathlib import Path

from aml.data.extract import extract_dataset
from aml.data.transform import transform_dataset
from aml.data.load import load_etl_result

def run_etl(source, output_dir, smoke=False):
    extracted = extract_dataset(source, smoke=smoke)

    transformed = transform_dataset(extracted.df)

    return load_etl_result(
        result=transformed,
        output_dir=output_dir,
        source_metadata=extracted,
        configuration={"smoke": smoke},
    )

def main():
    parser = argparse.ArgumentParser(description="Prepare AML data and save a completed ETL run")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_dir = run_etl(args.source, args.output, smoke=args.smoke)
    print(f"ETL completed: {run_dir}")

if __name__ == "__main__":
    main()