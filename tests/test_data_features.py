"""Extraction, validation, calendar periods, currency conversion and the assembled transform."""
import json

import numpy as np
import pandas as pd
import pytest

from aml.data.extract import calculate_sha256, extract_dataset, sample_senders
from aml.data.keys import account_key
from aml.data.split import SplitConfig, boundaries_from_starts, temporal_slices
from aml.data.transform import EtlConfig, transform_dataset
from aml.data.validate import validate_raw
from aml.features import CurrencyConverter, FeaturePipeline, build_basic_features
from conftest import EUR_TO_USD, make_transactions, write_ibm_csv

HOURLY = SplitConfig(unit="1h", warmup_units=1, val_units=1, threshold_units=1, test_size=0.2)


def test_csv_reads_ibm_header_keeps_identifiers_and_matches_parquet(tmp_path):
    raw = make_transactions(rows=500)
    raw.loc[0, "Account"] = "000123"
    csv = write_ibm_csv(raw, tmp_path / "raw.csv")
    parquet = tmp_path / "raw.parquet"
    raw.to_parquet(parquet, index=False)
    from_csv, from_parquet = extract_dataset(csv), extract_dataset(parquet)
    pd.testing.assert_frame_equal(validate_raw(from_csv.df).df, validate_raw(from_parquet.df).df)
    assert from_csv.df["Account"].iloc[0] == "000123"
    assert (from_csv.extracted_rows, from_csv.size_bytes) == (len(raw), csv.stat().st_size)
    assert from_csv.sha256 == calculate_sha256(csv)
    with pytest.raises(ValueError):
        calculate_sha256(csv, chunk_size=0)


def test_csv_rejects_unexpected_header_and_unparseable_values(tmp_path):
    raw = make_transactions(rows=20)
    bad_header = tmp_path / "header.csv"
    raw.drop(columns="Payment Format").to_csv(bad_header, index=False)
    with pytest.raises(ValueError, match="header"):
        extract_dataset(bad_header)
    bad_time = write_ibm_csv(raw, tmp_path / "time.csv")
    bad_time.write_text(bad_time.read_text().replace("2025/01/01", "01.01.2025", 1))
    with pytest.raises(Exception):
        extract_dataset(bad_time)
    with pytest.raises(FileNotFoundError):
        extract_dataset(tmp_path / "missing.csv")


def test_smoke_keeps_complete_sender_histories(tmp_path):
    raw = make_transactions(rows=2_000, accounts=400)
    csv = write_ibm_csv(raw, tmp_path / "raw.csv")
    full, smoke = extract_dataset(csv).df, extract_dataset(csv, smoke=True).df
    keep = account_key(full["From Bank"], full["Account"]) % 50 == 0
    pd.testing.assert_frame_equal(smoke, full.loc[keep].reset_index(drop=True))
    assert 0 < len(smoke) < len(full)
    pd.testing.assert_frame_equal(sample_senders(smoke), smoke)


def test_validation_report_and_quality_gates():
    raw = make_transactions(rows=200)
    result = validate_raw(pd.concat([raw, raw.iloc[:1]], ignore_index=True))
    report = json.loads(json.dumps(result.quality_report, allow_nan=False))
    assert report["duplicate_rows"] == 1
    assert report["positives"] + report["negatives"] == report["rows"] == 201
    assert sum(day["rows"] for day in report["daily"]) == 201
    assert result.df["Timestamp"].is_monotonic_increasing
    for column, value, message in [("Payment Format", "Telegraph", "payment formats"),
                                   ("Account", "", "Empty identifiers"),
                                   ("Amount Paid", -1.0, "Invalid amount"),
                                   ("Is Laundering", 2, "binary"),
                                   ("Account.1", None, "Missing values")]:
        broken = raw.copy()
        broken[column] = broken[column].astype(object)
        broken.loc[3, column] = value
        with pytest.raises(ValueError, match=message):
            validate_raw(broken)


def test_periods_follow_calendar_units_and_exclude_sparse_tail():
    days = pd.date_range("2025-01-01", periods=10 * 24, freq="h").repeat(10)
    tail = pd.date_range("2025-01-11", periods=3, freq="D")
    timestamps = pd.Series(days.append(tail))
    periods = temporal_slices(timestamps, SplitConfig(test_size=0.2))
    starts = {name: str(value) for name, value in periods.starts.items()}
    assert starts == {"warmup": "2025-01-01 00:00:00", "train": "2025-01-02 00:00:00",
                      "val": "2025-01-07 00:00:00", "threshold": "2025-01-08 00:00:00",
                      "test": "2025-01-09 00:00:00", "tail": "2025-01-11 00:00:00"}
    assert periods.tail == slice(2_400, 2_403)
    assert periods.warmup == slice(0, 240)
    kept = temporal_slices(timestamps, SplitConfig(test_size=0.2, tail_min_share=None))
    assert kept.tail == slice(2_403, 2_403) and "tail" not in kept.starts
    assert boundaries_from_starts(timestamps, periods.starts).slices() == periods.slices()


def test_period_planning_rejects_impossible_layouts():
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=4 * 24, freq="h"))
    with pytest.raises(ValueError, match="No training units"):
        temporal_slices(timestamps, SplitConfig(warmup_units=2, test_size=0.2))
    with pytest.raises(ValueError, match="sorted"):
        temporal_slices(timestamps.iloc[::-1])
    with pytest.raises(ValueError):
        SplitConfig(test_size=1.5)
    with pytest.raises(ValueError):
        SplitConfig(val_units=0)


def test_currency_rates_are_recovered_and_unknown_currencies_fail():
    raw = validate_raw(make_transactions(rows=2_000)).df
    converter = CurrencyConverter().fit(raw)
    assert np.exp(converter.log_rates_["Euro"]) == pytest.approx(EUR_TO_USD, rel=1e-3)
    usd = converter.transform(raw)
    expected = raw["Amount Paid"] * np.where(raw["Payment Currency"] == "Euro", EUR_TO_USD, 1.0)
    np.testing.assert_allclose(usd["amount_paid_usd"], expected, rtol=1e-3)
    np.testing.assert_allclose(usd["amount_paid_usd"], usd["amount_received_usd"], rtol=1e-2)
    unseen = raw.head(1).assign(**{"Payment Currency": "Yen"})
    with pytest.raises(ValueError, match="No exchange rate"):
        converter.transform(unseen)
    isolated = pd.concat([raw, raw.head(1).assign(**{"Payment Currency": "Yen",
                                                     "Receiving Currency": "Yen"})])
    with pytest.raises(ValueError, match="Cannot infer"):
        CurrencyConverter().fit(isolated)


def test_basic_features_exclude_calendar_dates():
    raw = validate_raw(make_transactions(rows=100)).df
    enriched = pd.concat([raw, CurrencyConverter().fit(raw).transform(raw)], axis=1)
    X = build_basic_features(enriched)
    assert not {"day", "month", "dayofweek", "dow_sin", "is_weekend", "Amount Paid"} & set(X)
    assert X["fx_log_deviation"].abs().median() < 1e-3
    tiny = pd.DataFrame({"amount_paid_usd": [0.01], "amount_received_usd": [0.011]})
    deviation = build_basic_features(enriched.head(1).drop(columns=list(tiny)).assign(**tiny.iloc[0]))
    assert deviation["fx_log_deviation"].iloc[0] == pytest.approx(np.log(1.1), rel=1e-4)


def test_transform_equals_one_pass_and_base_features_ignore_labels():
    raw = make_transactions(days=2, rows=1_500)
    config = EtlConfig(split=HOURLY)
    reversed_input = raw.iloc[::-1]
    result = transform_dataset(reversed_input, config)
    clean = validate_raw(reversed_input).df
    periods = temporal_slices(clean["Timestamp"], HOURLY)
    # Exchange rates come from warmup + train; later periods arrive as one batch here.
    pipeline = FeaturePipeline(config.windows)
    history = pipeline.fit_transform(clean.iloc[: periods.train.stop])
    later = pipeline.transform_update(clean.iloc[periods.val.start: periods.test.stop])
    encoded_train = result.encoder.fit_transform(
        history.iloc[periods.warmup.stop:], result.splits["train"].y,
        result.splits["train"].timestamps,
    )
    pd.testing.assert_frame_equal(
        pd.concat([split.X for split in result.splits.values()], ignore_index=True),
        pd.concat([encoded_train, result.encoder.transform(later)], ignore_index=True),
    )
    pd.testing.assert_series_equal(
        pd.concat([split.y for split in result.splits.values()], ignore_index=True),
        clean["Is Laundering"].iloc[periods.train.start: periods.test.stop].reset_index(drop=True),
    )
    assert result.metadata["excluded_rows"]["warmup"] == periods.warmup.stop
    json.dumps(result.metadata, allow_nan=False)

    flipped = transform_dataset(clean.assign(**{"Is Laundering": 1 - clean["Is Laundering"]}), config)
    base_columns = history.select_dtypes(include="number").columns
    for name, split in result.splits.items():
        pd.testing.assert_frame_equal(split.X[base_columns], flipped.splits[name].X[base_columns])


def test_etl_encoder_uses_train_labels_only_and_excludes_timestamp_ties():
    raw = validate_raw(make_transactions(days=2, rows=1_500)).df
    config = EtlConfig(split=HOURLY)
    periods = temporal_slices(raw["Timestamp"], HOURLY)
    baseline = transform_dataset(raw, config)
    changed = raw.copy()
    # Warmup and held-out labels must never enter the encoder's statistics.
    outside_train = ~changed.index.isin(range(periods.train.start, periods.train.stop))
    changed.loc[outside_train, "Is Laundering"] = 1 - changed.loc[outside_train, "Is Laundering"]
    other = transform_dataset(changed, config)
    for name in baseline.splits:
        pd.testing.assert_frame_equal(baseline.splits[name].X, other.splits[name].X)
    assert baseline.encoder.global_prior_ == pytest.approx(baseline.splits["train"].y.mean())

    cutoff = baseline.splits["train"].timestamps.iloc[len(baseline.splits["train"].X) // 2]
    changed = raw.copy()
    changed.loc[changed["Timestamp"] >= cutoff, "Is Laundering"] = (
        1 - changed.loc[changed["Timestamp"] >= cutoff, "Is Laundering"]
    )
    other = transform_dataset(changed, config)
    past = baseline.splits["train"].timestamps <= cutoff
    pd.testing.assert_frame_equal(baseline.splits["train"].X.loc[past], other.splits["train"].X.loc[past])
