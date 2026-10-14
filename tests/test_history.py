"""The incremental event-log history must equal a naive point-in-time recomputation."""
import pickle

import numpy as np
import pandas as pd
import pytest

from aml.data.schema import PAID_USD, RECEIVED_USD
from aml.features.historical import HistoricalFeatureBuilder, build_historical_features
from conftest import make_transactions

WINDOWS = {"10m": 600, "2h": 7_200}


def history_input(days=2, rows=600, accounts=12, seed=0):
    df = make_transactions(days=days, rows=rows, accounts=accounts, seed=seed)
    return df.assign(**{PAID_USD: df["Amount Paid"] * 1.5, RECEIVED_USD: df["Amount Received"] * 0.5})


def reference(df: pd.DataFrame, windows=WINDOWS) -> pd.DataFrame:
    """O(n^2) definition of every feature: use rows with a strictly earlier timestamp only."""
    t = df["Timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    sender = list(zip(df["From Bank"], df["Account"]))
    receiver = list(zip(df["To Bank"], df["Account.1"]))
    pair = list(zip(sender, receiver))
    paid, received = df[PAID_USD].to_numpy(), df[RECEIVED_USD].to_numpy()
    interbank = (df["From Bank"] != df["To Bank"]).to_numpy()
    first_seen = {}
    for p, time in zip(pair, t):
        first_seen.setdefault(p, time)

    def log_minutes(seconds):
        return -1.0 if seconds is None else np.log1p(seconds / 60)

    def to_mean(amount, values):
        mean = np.mean(values) if len(values) else 0.0
        return np.log1p(min(amount / mean, 1e6)) if mean > 0 else 0.0

    rows = []
    for i in range(len(df)):
        prior = np.flatnonzero(t < t[i])
        out = [j for j in prior if sender[j] == sender[i]]
        inc = [j for j in prior if receiver[j] == receiver[i]]
        out_times = sorted({t[j] for j in out})
        row = {
            "sender_prev_tx_log": np.log1p(len(out)),
            "receiver_prev_tx_log": np.log1p(len(inc)),
            "pair_prev_tx_log": np.log1p(sum(pair[j] == pair[i] for j in prior)),
            "is_new_pair": int(not any(pair[j] == pair[i] for j in prior)),
            "sender_unique_receivers_log": np.log1p(len({receiver[j] for j in out})),
            "receiver_unique_senders_log": np.log1p(len({sender[j] for j in inc})),
            "sender_prior_interbank_ratio": np.mean(interbank[out]) if out else 0.0,
            "sender_minutes_since_prev_log": log_minutes(t[i] - out_times[-1] if out else None),
            "receiver_minutes_since_prev_log": log_minutes(t[i] - max(t[inc]) if inc else None),
            "sender_prior_avg_minutes_log": log_minutes(
                np.mean(np.diff(out_times)) if len(out_times) > 1 else None),
            "amount_to_sender_prior_mean_log": to_mean(paid[i], paid[out]),
            "amount_to_receiver_prior_mean_log": to_mean(received[i], received[inc]),
        }
        for name, seconds in windows.items():
            def recent(rows_):
                return [j for j in rows_ if t[j] > t[i] - seconds]
            out_w, inc_w = recent(out), recent(inc)
            inflow = recent([j for j in prior if receiver[j] == sender[i]])
            inflow_sum = received[inflow].sum()
            outflow = paid[out_w].sum() + paid[i]
            new_receivers = {p[1] for p, time in first_seen.items()
                             if p[0] == sender[i] and t[i] - seconds < time < t[i]}
            new_senders = {p[0] for p, time in first_seen.items()
                           if p[1] == receiver[i] and t[i] - seconds < time < t[i]}
            row.update({
                f"sender_tx_{name}_log": np.log1p(len(out_w)),
                f"sender_amount_{name}_log": np.log1p(paid[out_w].sum()),
                f"receiver_tx_{name}_log": np.log1p(len(inc_w)),
                f"receiver_amount_{name}_log": np.log1p(received[inc_w].sum()),
                f"sender_new_receivers_{name}_log": np.log1p(len(new_receivers)),
                f"receiver_new_senders_{name}_log": np.log1p(len(new_senders)),
                f"sender_inflow_{name}_log": np.log1p(inflow_sum),
                f"sender_outflow_to_inflow_{name}_log":
                    np.log1p(min(outflow / inflow_sum, 1e6)) if inflow_sum > 0 else -1.0,
            })
        rows.append(row)
    return pd.DataFrame(rows)


def assert_features_equal(actual, expected):
    assert actual.columns.tolist() == expected.columns.tolist()
    np.testing.assert_allclose(actual.to_numpy(np.float64), expected.to_numpy(np.float64),
                               rtol=1e-5, atol=1e-5)


def batches(df, n_batches, seed):
    """Random cut points that never split a timestamp group."""
    starts = np.flatnonzero(df["Timestamp"].ne(df["Timestamp"].shift()).to_numpy())[1:]
    cuts = np.sort(np.random.default_rng(seed).choice(starts, n_batches - 1, replace=False))
    return [df.iloc[a:b] for a, b in zip([0, *cuts], [*cuts, len(df)])]


@pytest.fixture(scope="module")
def data():
    df = history_input()
    return df, reference(df)


def test_one_pass_matches_reference(data):
    df, expected = data
    actual = build_historical_features(df, df["Timestamp"], WINDOWS)
    assert_features_equal(actual, expected[actual.columns])
    assert actual["is_new_pair"].dtype == np.int8


@pytest.mark.parametrize("n_batches", [2, 7, 40])
def test_incremental_batches_match_reference(data, n_batches):
    # Windows are minutes long and data spans days: most history is folded into the archive.
    df, expected = data
    builder = HistoricalFeatureBuilder(WINDOWS)
    parts = batches(df, n_batches, seed=n_batches)
    results = [builder.fit_transform(parts[0], parts[0]["Timestamp"])]
    results += [builder.transform_update(part, part["Timestamp"]) for part in parts[1:]]
    assert_features_equal(pd.concat(results, ignore_index=True), expected[results[0].columns])


def test_history_state_stays_bounded(data):
    df, _ = data
    builder = HistoricalFeatureBuilder(WINDOWS)
    for i, part in enumerate(batches(df, 20, seed=1)):
        (builder.fit_transform if i == 0 else builder.transform_update)(part, part["Timestamp"])
        last = part["Timestamp"].iloc[-1]
        logged = df["Timestamp"].le(last) & df["Timestamp"].gt(last - pd.Timedelta("2h"))
        assert len(builder.outgoing_.keys) == logged.sum()
        assert len(builder.pairs_.keys) == 0


def test_transform_scores_without_remembering(data):
    df, _ = data
    head, tail = df.iloc[:300], df.iloc[300:]
    tail = tail[tail["Timestamp"] > head["Timestamp"].iloc[-1]]
    builder = HistoricalFeatureBuilder(WINDOWS).fit(head, head["Timestamp"])
    first = builder.transform(tail, tail["Timestamp"])
    pd.testing.assert_frame_equal(first, builder.transform(tail, tail["Timestamp"]))
    pd.testing.assert_frame_equal(first, builder.transform_update(tail, tail["Timestamp"]))
    with pytest.raises(ValueError, match="strictly after"):
        builder.transform(tail, tail["Timestamp"])


def test_features_ignore_future_rows_and_order_within_timestamp(data):
    df, _ = data
    actual = build_historical_features(df, df["Timestamp"], WINDOWS)
    cut = 250
    while df["Timestamp"].iloc[cut] == df["Timestamp"].iloc[cut - 1]:
        cut += 1
    changed = df.copy()
    changed.loc[cut:, [PAID_USD, RECEIVED_USD]] = 1e9
    changed.loc[cut:, "Account"] = "000001"
    pd.testing.assert_frame_equal(
        actual.iloc[:cut], build_historical_features(changed, changed["Timestamp"], WINDOWS).iloc[:cut]
    )
    order = np.lexsort((np.random.default_rng(3).random(len(df)), df["Timestamp"].to_numpy()))
    shuffled = build_historical_features(df.iloc[order], df["Timestamp"].iloc[order], WINDOWS)
    pd.testing.assert_frame_equal(shuffled, actual.iloc[order].reset_index(drop=True))


def test_builder_survives_pickling(data):
    df, _ = data
    head, tail = df.iloc[:300], df.iloc[300:]
    tail = tail[tail["Timestamp"] > head["Timestamp"].iloc[-1]]
    builder = HistoricalFeatureBuilder(WINDOWS).fit(head, head["Timestamp"])
    restored = pickle.loads(pickle.dumps(builder))
    pd.testing.assert_frame_equal(restored.transform(tail, tail["Timestamp"]),
                                  builder.transform(tail, tail["Timestamp"]))


def test_small_window_sum_survives_large_neighbour():
    df = history_input(rows=4)
    df["Timestamp"] = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:01",
                                      "2025-01-01 00:02", "2025-01-01 00:03"])
    df[["From Bank", "Account"]] = ["001", "000001"]
    df[PAID_USD] = [1e20, 1e20, 1.0, 1.0]
    actual = build_historical_features(df, df["Timestamp"], {"90s": 90})
    assert actual["sender_amount_90s_log"].iloc[3] == pytest.approx(np.log1p(1.0))


def test_builder_validates_inputs(data):
    df, _ = data
    builder = HistoricalFeatureBuilder(WINDOWS)
    with pytest.raises(RuntimeError):
        builder.transform(df, df["Timestamp"])
    with pytest.raises(ValueError, match="sorted"):
        builder.fit(df, df["Timestamp"].iloc[::-1])
    with pytest.raises(ValueError, match="same number of rows"):
        builder.fit(df, df["Timestamp"].iloc[:-1])
    with pytest.raises(ValueError, match="Missing history columns"):
        builder.fit(df.drop(columns=PAID_USD), df["Timestamp"])
    builder.fit(df.iloc[:1], df["Timestamp"].iloc[:1])
    same_time = df["Timestamp"] == df["Timestamp"].iloc[0]
    with pytest.raises(ValueError, match="strictly after"):
        builder.transform(df[same_time], df.loc[same_time, "Timestamp"])
    with pytest.raises(ValueError, match="positive"):
        HistoricalFeatureBuilder({"bad": 0})
