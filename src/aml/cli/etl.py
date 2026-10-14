from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from aml.data.extract import extract_dataset
from aml.data.load import load_etl_result
from aml.data.transform import EtlConfig, transform_dataset


def run_etl(source, output_dir, smoke: bool = False, config: EtlConfig = EtlConfig()) -> Path:
    extracted = extract_dataset(source, smoke=smoke)
    transformed = transform_dataset(extracted.df, config)
    return load_etl_result(
        result=transformed,
        output_dir=output_dir,
        source_metadata=extracted,
        configuration={"smoke": smoke, **config.to_dict()},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare AML data and save a completed ETL run")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--smoke", action="store_true",
                        help="keep complete histories of every 50th sender account")
    parser.add_argument("--warmup-days", type=int, default=EtlConfig().split.warmup_units,
                        help="leading days used only as account history")
    parser.add_argument("--keep-sparse-tail", action="store_true",
                        help="do not exclude trailing days with almost no traffic")
    args = parser.parse_args(argv)
    default = EtlConfig()
    split = replace(default.split, warmup_units=args.warmup_days,
                    tail_min_share=None if args.keep_sparse_tail else default.split.tail_min_share)
    run_dir = run_etl(args.source, args.output, smoke=args.smoke, config=replace(default, split=split))
    print(f"ETL completed: {run_dir}")


if __name__ == "__main__":
    main()
