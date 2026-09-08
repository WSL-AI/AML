"""Public loading helpers shared with the notebook workflow."""
from pathlib import Path

import pandas as pd

from aml.workflow import RAW_COLUMNS, TARGET, validate_raw, resolve_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]
PATH = PROJECT_ROOT / 'data/processed'
SPLIT_PATH = PROJECT_ROOT / 'artifacts/runs/full/data'
CSV_COLUMNS = RAW_COLUMNS


def load_raw_data(file_name='raw', path=PATH, dtypes=None):
    """Read validated raw Parquet; retain identical rows without a transaction ID."""
    df = pd.read_parquet(Path(path) / f'{file_name}.parquet', columns=RAW_COLUMNS)
    if dtypes is not None:
        df = df.astype(dtypes)
    return validate_raw(df)


def load_dataset_splits(sets=None, input_dir=None, target_col=TARGET):
    """Load explicit splits deterministically; missing files are errors, not omissions.

    None loads train/val/threshold/test. 'all' discovers labeled parquet files,
    excluding the timestamp sidecars. Raw data is included only if present and
    explicitly requested with 'all' or 'raw'.
    """
    path = Path(input_dir) if input_dir is not None else resolve_run(PROJECT_ROOT) / "data"
    if not path.is_dir():
        raise FileNotFoundError(path)
    if sets is None:
        names = ['train', 'val', 'threshold', 'test']
    elif sets == 'all' or sets == ['all']:
        names = sorted(p.stem for p in path.glob('*.parquet') if not p.stem.endswith('_time'))
    else:
        names = [sets] if isinstance(sets, str) else list(sets)
    if not names:
        raise FileNotFoundError(f'No dataset splits in {path}')
    result = {}
    for name in names:
        if Path(name).name != name:
            raise ValueError('Expected a split name, not a path')
        frame = pd.read_parquet(path / f'{name}.parquet')
        if target_col not in frame:
            raise ValueError(f'{name}: missing target {target_col}')
        result[name] = (frame.drop(columns=[target_col]), frame[target_col])
    return result
