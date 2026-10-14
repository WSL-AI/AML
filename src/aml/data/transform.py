from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd

from aml.features.historical import DEFAULT_WINDOWS
from aml.features.encoding import OrderedTargetEncoder
from aml.features.pipeline import FeaturePipeline

from .schema import CATEGORICAL, TARGET, TIMESTAMP
from .split import MODEL_PERIODS, SplitConfig, temporal_slices, validate_split_boundaries
from .validate import validate_raw


@dataclass(frozen=True)
class EtlConfig:
    split: SplitConfig = field(default_factory=SplitConfig)
    windows: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_WINDOWS))
    anchor_currency: str = "US Dollar"
    encoding_smoothing: tuple[float, ...] = (20.0, 200.0)
    encoding_interactions: tuple[tuple[str, ...], ...] = (
        ("From Bank", "Account"),
        ("To Bank", "Account.1"),
        ("From Bank", "To Bank"),
        ("Receiving Currency", "Payment Currency"),
    )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetSplit:
    X: pd.DataFrame
    y: pd.Series
    timestamps: pd.Series


@dataclass
class TransformResult:
    splits: dict[str, DatasetSplit]
    pipeline: FeaturePipeline
    encoder: OrderedTargetEncoder
    feature_names: list[str]
    feature_schema: dict[str, str]
    metadata: dict


def transform_dataset(df: pd.DataFrame, config: EtlConfig = EtlConfig()) -> TransformResult:
    """Validate, cut calendar periods and build point-in-time features.

    Warmup rows only seed account histories; tail rows are excluded entirely.
    Each later period is scored after every earlier period has entered the history.
    """
    validated = validate_raw(df)
    df = validated.df
    timestamps = df[TIMESTAMP]
    boundaries = temporal_slices(timestamps, config.split)
    period_report = validate_split_boundaries(timestamps, boundaries)

    pipeline = FeaturePipeline(config.windows, config.anchor_currency)
    warmup_rows = boundaries.warmup.stop
    history = pipeline.fit_transform(df.iloc[: boundaries.train.stop])
    features = {"train": history.iloc[warmup_rows:].reset_index(drop=True)}
    for name in MODEL_PERIODS[1:]:
        features[name] = pipeline.transform_update(df.iloc[getattr(boundaries, name)])

    splits = {
        name: DatasetSplit(
            X=features[name],
            y=df[TARGET].iloc[getattr(boundaries, name)].reset_index(drop=True),
            timestamps=timestamps.iloc[getattr(boundaries, name)].reset_index(drop=True),
        )
        for name in MODEL_PERIODS
    }
    encoder = OrderedTargetEncoder(
        CATEGORICAL, interactions=config.encoding_interactions,
        smoothing=config.encoding_smoothing,
    )
    train = splits["train"]
    train.X = encoder.fit_transform(train.X, train.y, train.timestamps)
    for name in MODEL_PERIODS[1:]:
        splits[name].X = encoder.transform(splits[name].X)
    train_X = train.X
    feature_names = train_X.columns.tolist()
    for name, split in splits.items():
        if split.X.columns.tolist() != feature_names or not split.X.dtypes.equals(train_X.dtypes):
            raise ValueError(f"{name}: inconsistent feature schema")

    for name, part in boundaries.slices().items():
        period_report[name]["positives"] = int(df[TARGET].iloc[part].sum())
    return TransformResult(
        splits=splits,
        pipeline=pipeline,
        encoder=encoder,
        feature_names=feature_names,
        feature_schema={name: str(dtype) for name, dtype in train_X.dtypes.items()},
        metadata={
            "rows": len(df),
            "history_mode": "sequential",
            "encoding": {
                "method": "ordered_target",
                "fit_period": "train",
                "categorical_features": list(CATEGORICAL),
                "interactions": [list(spec) for spec in config.encoding_interactions],
                "smoothing": list(config.encoding_smoothing),
                "add_log_count": encoder.add_log_count,
                "initial_prior": encoder.initial_prior,
                "label_assumption": "labels available immediately; offline only",
            },
            "config": config.to_dict(),
            "period_starts": {name: start.isoformat() for name, start in boundaries.starts.items()},
            "periods": period_report,
            "excluded_rows": {"warmup": boundaries.warmup.stop - boundaries.warmup.start,
                              "tail": boundaries.tail.stop - boundaries.tail.start},
            "currency": pipeline.converter_.report(),
            "quality_report": validated.quality_report,
        },
    )
