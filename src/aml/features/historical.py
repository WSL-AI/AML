from __future__ import annotations

import numpy as np
import pandas as pd

from aml.data.schema import TARGET


class HistoricalFeatureBuilder:
    @staticmethod
    def _prepare(X, timestamps):
        X = X.reset_index(drop=True).copy()
        timestamps = pd.to_datetime(pd.Series(timestamps), errors="raise").reset_index(drop=True)
        if X.empty or len(X) != len(timestamps):
            raise ValueError("Expected a nonempty batch with matching timestamps")
        if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
            raise ValueError("Expected sorted timestamps without NaT")
        if not X.columns.is_unique or TARGET in X.columns:
            raise ValueError("Features must have unique columns and exclude the target")
        return X, timestamps

    def fit(self, X, timestamps):
        X, timestamps = self._prepare(X, timestamps)
        self.history_X_ = X
        self.history_time_ = timestamps
        self.feature_columns_ = X.columns.tolist()

        return self

    def fit_transform(self, X, timestamps):
        X, timestamps = self._prepare(X, timestamps)
        features = build_historical_features(X, timestamps)
        self.fit(X, timestamps)
        return features

    def _transform(self, X, timestamps):
        if not hasattr(self, "history_X_"):
            raise RuntimeError("Call fit or fit_transform first")

        X, timestamps = self._prepare(X, timestamps)
        if X.columns.tolist() != self.feature_columns_:
            raise ValueError("Feature columns differ from fitted history")
        if not X.dtypes.equals(self.history_X_.dtypes):
            raise ValueError("Feature dtypes differ from fitted history")
        if timestamps.iloc[0] <= self.history_time_.iloc[-1]:
            raise ValueError("New batch must be strictly after saved history; keep timestamp groups together")
        history_rows = len(self.history_X_)

        combined_X = pd.concat([self.history_X_, X], ignore_index=True)
        combined_time = pd.concat([self.history_time_, timestamps], ignore_index=True)
        combined_features = build_historical_features(combined_X, combined_time)
        new_features = combined_features.iloc[history_rows:].reset_index(drop=True)

        return new_features, combined_X, combined_time

    def transform(self, X, timestamps):
        features, _, _ = self._transform(X, timestamps)
        return features

    def transform_update(self, X, timestamps):
        features, combined_X, combined_time = self._transform(X, timestamps)
        self.history_X_ = combined_X
        self.history_time_ = combined_time

        return features
       


def build_historical_features(
    X: pd.DataFrame,
    timestamp: pd.Series,
) -> pd.DataFrame:
    X = X.reset_index(drop=True).copy()
    timestamp = pd.to_datetime(pd.Series(timestamp), errors="raise").reset_index(drop=True)

    if len(X) != len(timestamp):
        raise ValueError("X and timestamp must have the same number of rows")
    if timestamp.isna().any() or not timestamp.is_monotonic_increasing:
        raise ValueError("timestamp must be sorted in ascending order")

    # Equal timestamps form one information batch: no row may see another row in the same batch.
    time_group = timestamp.ne(timestamp.shift()).cumsum().astype("int32")

    # Account numbers are treated as bank-scoped identities.
    sender = _composite_key(X, ["From Bank", "Account"]).reset_index(drop=True)
    receiver = _composite_key(X, ["To Bank", "Account.1"]).reset_index(drop=True)
    pair = pd.util.hash_pandas_object(
        pd.DataFrame({"sender": sender, "receiver": receiver}),
        index=False,
    ).astype("uint64")

    paid = X["Amount Paid"].clip(lower=0).reset_index(drop=True)

    # Transaction counts from strictly earlier timestamps.
    sender_previous_count = _strict_prior_count(sender, time_group)
    receiver_previous_count = _strict_prior_count(receiver, time_group)
    pair_previous_count = _strict_prior_count(pair, time_group)

    X["sender_prev_tx_log"] = np.log1p(sender_previous_count).astype("float32")
    X["receiver_prev_tx_log"] = np.log1p(receiver_previous_count).astype("float32")
    X["pair_prev_tx_log"] = np.log1p(pair_previous_count).astype("float32")
    X["is_new_pair"] = (pair_previous_count == 0).astype("int8")

    # Unique counterparties established strictly before the current timestamp.
    first_pair_row = pair.groupby(pair, sort=False).cumcount().eq(0).astype("int32")
    sender_unique_receivers_before = _strict_prior_cumsum(
        first_pair_row, sender, time_group
    )
    receiver_unique_senders_before = _strict_prior_cumsum(
        first_pair_row, receiver, time_group
    )
    X["sender_unique_receivers_log"] = np.log1p(
        sender_unique_receivers_before
    ).astype("float32")
    X["receiver_unique_senders_log"] = np.log1p(
        receiver_unique_senders_before
    ).astype("float32")

    # Historical interbank share.
    is_interbank = (
        X["From Bank"] != X["To Bank"]
    ).astype("int32").reset_index(drop=True)
    sender_prior_interbank_count = _strict_prior_cumsum(
        is_interbank, sender, time_group
    )
    X["sender_prior_interbank_ratio"] = (
        sender_prior_interbank_count
        / sender_previous_count.replace(0, np.nan)
    ).fillna(0).astype("float32")

    # Time since the previous distinct activity timestamp.
    sender_previous_timestamp = _strict_previous_timestamp(timestamp, sender)
    receiver_previous_timestamp = _strict_previous_timestamp(timestamp, receiver)

    sender_minutes_since_previous = (
        (timestamp - sender_previous_timestamp).dt.total_seconds() / 60
    )
    receiver_minutes_since_previous = (
        (timestamp - receiver_previous_timestamp).dt.total_seconds() / 60
    )
    X["sender_minutes_since_prev_log"] = np.log1p(
        sender_minutes_since_previous.clip(lower=0)
    ).fillna(-1).astype("float32")
    X["receiver_minutes_since_prev_log"] = np.log1p(
        receiver_minutes_since_previous.clip(lower=0)
    ).fillna(-1).astype("float32")

    # Average interval between distinct historical sender activity timestamps.
    # The current timestamp's interval is excluded, and the denominator counts
    # actual historical intervals rather than historical transactions.
    sender_previous_row_timestamp = timestamp.groupby(sender, sort=False).shift()
    sender_batch_start = sender_previous_row_timestamp.ne(timestamp)
    sender_interval_minutes = sender_minutes_since_previous.where(sender_batch_start)

    sender_prior_interval_sum = _strict_prior_cumsum(
        sender_interval_minutes.fillna(0.0),
        sender,
        time_group,
    )
    sender_prior_interval_count = _strict_prior_cumsum(
        sender_interval_minutes.notna().astype("int32"),
        sender,
        time_group,
    )
    sender_prior_average_minutes = (
        sender_prior_interval_sum
        / sender_prior_interval_count.replace(0, np.nan)
    )
    X["sender_prior_avg_minutes_log"] = np.log1p(
        sender_prior_average_minutes.clip(lower=0)
    ).fillna(-1).astype("float32")

    # Amount anomaly relative to the sender's strictly historical mean amount.
    sender_currency = _composite_key(X, ["From Bank", "Account", "Payment Currency"])
    sender_currency_count = _strict_prior_count(sender_currency, time_group)
    sender_prior_amount_sum = _strict_prior_cumsum(paid, sender_currency, time_group)
    sender_prior_amount_mean = (
        sender_prior_amount_sum
        / sender_currency_count.replace(0, np.nan)
    )
    amount_to_prior_mean = paid / sender_prior_amount_mean.replace(0, np.nan)
    X["amount_to_sender_prior_mean_log"] = (
        np.log1p(amount_to_prior_mean.clip(lower=0, upper=1e6))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
    )

    return X

def _composite_key(X: pd.DataFrame, columns) -> pd.Series:
    return pd.util.hash_pandas_object(X.loc[:, list(columns)], index=False).astype("uint64")


def _strict_prior_count(entity: pd.Series, time_group: pd.Series) -> pd.Series:
    before = entity.groupby(entity, sort=False).cumcount()
    same_time = entity.groupby([entity, time_group], sort=False).cumcount()
    return (before - same_time).astype("int32")


def _strict_prior_cumsum(values, entity, time_group) -> pd.Series:
    # Shift timestamp batch totals: subtracting large cumulative sums can erase
    # a small historical value through floating-point cancellation.
    batches = pd.DataFrame({"entity": entity, "time": time_group, "value": values})
    totals = batches.groupby(["entity", "time"], sort=False)["value"].sum()
    cumulative = totals.groupby(level="entity", sort=False).cumsum()
    prior = cumulative.groupby(level="entity", sort=False).shift(fill_value=0)
    keys = pd.MultiIndex.from_arrays([entity, time_group], names=["entity", "time"])
    return pd.Series(prior.reindex(keys).to_numpy(), index=values.index)


def _strict_previous_timestamp(timestamp, entity) -> pd.Series:
    previous = timestamp.groupby(entity, sort=False).shift()
    return previous.where(previous < timestamp).groupby(entity, sort=False).ffill()

historical_features = build_historical_features
