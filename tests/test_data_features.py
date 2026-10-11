"""Regression checks for ETL metadata, positional splits and information boundaries."""
import json

import numpy as np
import pandas as pd
import pytest

from aml.data.extract import extract_dataset, calculate_sha256
from aml.data.split import SplitBoundaries, temporal_slices, validate_split_boundaries
from aml.data.validate import validate_raw
from aml.features import build_basic_features, build_historical_features
from aml.features.historical import _strict_prior_cumsum


def transactions():
    n = 12
    return pd.DataFrame({
        "Timestamp": pd.to_datetime(["2025-01-01 00:00"] * 2 + ["2025-01-01 00:01"] * 2
                                     + [f"2025-01-01 00:{i:02d}" for i in range(2, 10)]),
        "From Bank": ["001"] * n, "Account": ["000123"] * n,
        "To Bank": ["002"] * n, "Account.1": ["000456"] * n,
        "Payment Currency": ["USD"] * n, "Receiving Currency": ["USD"] * n,
        "Payment Format": ["Wire"] * n,
        "Amount Paid": np.arange(1, n + 1, dtype=float),
        "Amount Received": np.arange(1, n + 1, dtype=float),
        "Is Laundering": [0, 1] * 6,
    })


def features(df):
    clean = validate_raw(df).df
    return build_historical_features(build_basic_features(clean), clean.Timestamp)


def test_csv_preserves_identifiers_and_matches_parquet(tmp_path):
    raw = transactions()
    csv, parquet = tmp_path / "raw.csv", tmp_path / "raw.parquet"
    raw.to_csv(csv, index=False)
    raw.to_parquet(parquet, index=False)
    from_csv = extract_dataset(csv)
    from_parquet = extract_dataset(parquet)
    pd.testing.assert_frame_equal(validate_raw(from_csv.df).df, validate_raw(from_parquet.df).df)
    assert from_csv.df.Account.iloc[0] == "000123"
    assert from_csv.extracted_rows == len(raw)
    assert from_csv.size_bytes == csv.stat().st_size
    assert from_csv.sha256 == calculate_sha256(csv)
    for path in (csv, parquet):
        pd.testing.assert_frame_equal(extract_dataset(path, smoke=True).df,
                                      extract_dataset(path).df.iloc[::50].reset_index(drop=True))
    with pytest.raises(ValueError):
        calculate_sha256(csv, chunk_size=0)


def test_quality_report_is_json_serializable_and_counts_amounts():
    raw = transactions()
    result = validate_raw(pd.concat([raw, raw.iloc[:1]], ignore_index=True))
    report = json.loads(json.dumps(result.quality_report))
    assert report["duplicate_rows"] == 1
    assert report["negative_amounts"] == 0
    assert report["negatives"] == 7
    assert report["positives"] == 6
    assert sum(report["missing_values"].values()) == 0
    assert len(result.df) == 13


def test_splits_are_slices_and_keep_timestamp_batches_together():
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=19).insert(13, pd.Timestamp("2025-01-14")))
    boundaries = temporal_slices(timestamps)
    report = validate_split_boundaries(timestamps, boundaries)
    assert boundaries.train == slice(0, 13)
    assert sum(period["rows"] for period in report.values()) == len(timestamps)
    with pytest.raises(ValueError):
        temporal_slices(timestamps.iloc[::-1])
    with pytest.raises(ValueError):
        temporal_slices(timestamps, train_size=0.95)
    with pytest.raises(ValueError):
        temporal_slices(pd.Series(pd.to_datetime(["2025-01-01"] * 20)))
    with pytest.raises(ValueError):
        validate_split_boundaries(timestamps, SplitBoundaries(slice(0, 14), slice(14, 18),
                                                              slice(18, 19), slice(19, 20)))


def test_history_excludes_equal_timestamps_and_future():
    raw = transactions()
    actual = features(raw)
    np.testing.assert_allclose(actual.sender_prev_tx_log[:4], np.log1p([0, 0, 2, 2]))
    np.testing.assert_allclose(actual.sender_unique_receivers_log[:4], np.log1p([0, 0, 1, 1]))
    np.testing.assert_allclose(actual.sender_minutes_since_prev_log[2:4], np.log1p([1, 1]))
    assert actual.sender_prior_avg_minutes_log.iloc[2] == -1
    assert actual.sender_prior_avg_minutes_log.iloc[4] == pytest.approx(np.log1p(1))
    changed = raw.copy()
    changed.loc[4:, ["Amount Paid", "Amount Received"]] = 1e9
    changed.loc[4:, "Is Laundering"] = 1
    pd.testing.assert_frame_equal(actual.iloc[:4], features(changed).iloc[:4])
    permuted = raw.iloc[[1, 0, 3, 2, *range(4, len(raw))]]
    pd.testing.assert_frame_equal(actual.iloc[4:], features(permuted).iloc[4:])
    assert "Timestamp" not in actual and "Is Laundering" not in actual


def test_bank_identity_currency_and_input_preservation():
    raw = transactions()
    raw.loc[2, "From Bank"] = "OTHER"
    raw.loc[3, "Payment Currency"] = "EUR"
    original = raw.copy(deep=True)
    result = features(raw)
    assert result.loc[2, "sender_prev_tx_log"] == 0
    assert result.loc[3, "amount_to_sender_prior_mean_log"] == 0
    assert result.loc[3, "amount_log_gap"] == 0
    assert result.loc[4, "amount_to_sender_prior_mean_log"] == pytest.approx(np.log1p(5 / 1.5))
    pd.testing.assert_frame_equal(raw, original)
    basic = build_basic_features(raw)
    with pytest.raises(ValueError):
        build_historical_features(basic, raw.Timestamp.iloc[:-1])
    with pytest.raises(ValueError):
        build_historical_features(basic, raw.Timestamp.iloc[::-1])


def test_small_prior_sum_survives_large_current_batch():
    result = _strict_prior_cumsum(pd.Series([1., 1e20, 2.]),
                                 pd.Series(["a"] * 3), pd.Series([0, 1, 1]))
    np.testing.assert_array_equal(result, [0., 1., 1.])


def test_legacy_feature_imports_work():
    from aml.features.basic import basic_features
    from aml.features.historical import historical_features
    assert basic_features is build_basic_features
    assert historical_features is build_historical_features


def test_transform_matches_whole_history_and_keeps_rows_aligned():
    from aml.data.transform import transform_dataset
    raw = transactions().iloc[::-1].copy()
    original = raw.copy(deep=True)
    result = transform_dataset(raw)
    clean = validate_raw(raw).df
    pd.testing.assert_frame_equal(
        pd.concat([split.X for split in result.splits.values()], ignore_index=True), features(raw)
    )
    pd.testing.assert_series_equal(
        pd.concat([split.y for split in result.splits.values()], ignore_index=True), clean["Is Laundering"]
    )
    pd.testing.assert_series_equal(
        pd.concat([split.timestamps for split in result.splits.values()], ignore_index=True), clean.Timestamp
    )
    assert result.encoder is None
    assert result.feature_names == result.splits["train"].X.columns.tolist()
    json.dumps(result.metadata)
    pd.testing.assert_frame_equal(raw, original)
    changed = clean.copy()
    changed["Is Laundering"] = 1 - changed["Is Laundering"]
    changed.loc[10:, ["Amount Paid", "Amount Received"]] = 1e9
    other = transform_dataset(changed)
    for name in ("train", "val", "threshold"):
        pd.testing.assert_frame_equal(result.splits[name].X, other.splits[name].X)


def test_builder_keeps_copy_and_updates_only_when_requested():
    from aml.features.historical import HistoricalFeatureBuilder
    raw = validate_raw(transactions()).df
    X = build_basic_features(raw)
    train = X.iloc[:4].copy()
    time = raw.Timestamp.iloc[:4].copy()
    builder = HistoricalFeatureBuilder().fit(train, time)
    train.iloc[0, train.columns.get_loc("Amount Paid")] = 999
    time.iloc[0] = pd.Timestamp("2030-01-01")
    pd.testing.assert_frame_equal(builder.history_X_, X.iloc[:4].reset_index(drop=True))
    batch, batch_time = X.iloc[4:6], raw.Timestamp.iloc[4:6]
    first = builder.transform(batch, batch_time)
    pd.testing.assert_frame_equal(first, builder.transform(batch, batch_time))
    assert len(builder.history_X_) == 4
    pd.testing.assert_frame_equal(first, builder.transform_update(batch, batch_time))
    assert len(builder.history_X_) == 6
    with pytest.raises(ValueError, match="strictly after"):
        builder.transform_update(batch, batch_time)


def test_builder_rejects_target_schema_mismatch_and_split_timestamp_groups():
    from aml.features.historical import HistoricalFeatureBuilder
    raw = validate_raw(transactions()).df
    X = build_basic_features(raw)
    builder = HistoricalFeatureBuilder()
    with pytest.raises(RuntimeError):
        builder.transform(X, raw.Timestamp)
    with pytest.raises(ValueError, match="exclude the target"):
        builder.fit(X.assign(**{"Is Laundering": 0}), raw.Timestamp)
    builder.fit(X.iloc[:1], raw.Timestamp.iloc[:1])
    with pytest.raises(ValueError, match="strictly after"):
        builder.transform(X.iloc[1:2], raw.Timestamp.iloc[1:2])
    builder.fit(X.iloc[:4], raw.Timestamp.iloc[:4])
    with pytest.raises(ValueError, match="columns differ"):
        builder.transform(X.iloc[4:].drop(columns="hour"), raw.Timestamp.iloc[4:])
