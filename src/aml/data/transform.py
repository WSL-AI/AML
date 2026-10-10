from __future__ import annotations

import pandas as pd

from aml.features.basic import build_basic_features
from aml.features.historical import HistoricalFeatureBuilder

from .split import temporal_slices, validate_split_boundaries
from .validate import validate_raw

from .schema import TARGET, TIMESTAMP

from dataclasses import dataclass

@dataclass
class DatasetSplit:
    X: pd.DataFrame
    y: pd.Series
    timestamps: pd.Series

@dataclass
class TransformResult:
    splits: dict[str, DatasetSplit]
    encoder: object | None
    feature_names: list[str]
    feature_schema: dict[str, str]
    metadata: dict

def transform_dataset(df: pd.DataFrame) -> TransformResult:
    validated = validate_raw(df)
    df = validated.df
    timestamps = df[TIMESTAMP]
    boundaries = temporal_slices(timestamps)
    split_report = validate_split_boundaries(timestamps, boundaries)
    basic = build_basic_features(df)

    builder = HistoricalFeatureBuilder()

    splits = {}
    for name in ("train", "val", "threshold", "test"):
        partition = getattr(boundaries, name)
        X = basic.iloc[partition].reset_index(drop=True)
        time = timestamps.iloc[partition].reset_index(drop=True)
        if name == "train":
            X = builder.fit_transform(X, time)
        else:
            X = builder.transform_update(X, time)

        splits[name] = DatasetSplit(
            X=X,
            y=df[TARGET].iloc[partition].reset_index(drop=True).copy(),
            timestamps=time,
        )

    train_X = splits["train"].X
    feature_names = train_X.columns.tolist()
    feature_schema = {name: str(dtype) for name, dtype in train_X.dtypes.items()}
    for name, split in splits.items():
        if split.X.columns.tolist() != feature_names or not split.X.dtypes.equals(train_X.dtypes):
            raise ValueError(f"{name}: inconsistent feature schema")

    return TransformResult(
        splits=splits,
        encoder=None,
        feature_names=feature_names,
        feature_schema=feature_schema,
        metadata={
            "rows": len(df),
            "history_mode": "sequential",
            "encoding": None,
            "splits": split_report,
            "quality_report": validated.quality_report,
        },
    )