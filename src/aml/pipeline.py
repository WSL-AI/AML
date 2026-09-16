from pathlib import Path
import json, uuid
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from aml.data.load import CATEGORICAL, TARGET

def ranking_metrics(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    if len(y) != len(scores) or not len(y) or not np.isfinite(scores).all():
        raise ValueError("Invalid score vector")
    return {"AP": float(average_precision_score(y, scores)) if np.any(y == 1) else float("nan"),
            "ROC-AUC": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan")}

def lightgbm_validation(X, y):
    """Use the preferred API on new LightGBM and retain compatibility with 4.x."""
    import inspect
    from lightgbm import LGBMClassifier
    if "eval_X" in inspect.signature(LGBMClassifier.fit).parameters:
        return {"eval_X": X, "eval_y": y}
    return {"eval_set": [(X, y)]}

def basic_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[*CATEGORICAL, "Amount Paid", "Amount Received"]].reset_index(drop=True).copy()
    timestamp = df["Timestamp"].reset_index(drop=True)

    # Time
    hour = timestamp.dt.hour
    day_of_week = timestamp.dt.dayofweek
    paid = df["Amount Paid"].clip(lower=0).reset_index(drop=True)
    received = df["Amount Received"].clip(lower=0).reset_index(drop=True)

    X["hour"] = hour.astype("int8")
    X["dayofweek"] = day_of_week.astype("int8")
    X["day"] = timestamp.dt.day.astype("int8")
    X["month"] = timestamp.dt.month.astype("int8")

    X["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    X["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7).astype("float32")
    X["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7).astype("float32")
    X["is_night"] = hour.between(0, 5).astype("int8")
    X["is_weekend"] = day_of_week.isin([5, 6]).astype("int8")

    # Current-transaction consistency
    X["is_currency_same"] = (
        df["Receiving Currency"] == df["Payment Currency"]
    ).astype("int8").to_numpy()
    X["is_same_bank"] = (
        df["From Bank"] == df["To Bank"]
    ).astype("int8").to_numpy()
    X["is_self_transfer"] = (
        (df["From Bank"] == df["To Bank"])
        & (df["Account"] == df["Account.1"])
    ).astype("int8").to_numpy()

    # Amount transformations
    X["log_amount_paid"] = np.log1p(paid).astype("float32")
    X["log_amount_received"] = np.log1p(received).astype("float32")
    X["amount_log_gap"] = np.abs(
        X["log_amount_paid"] - X["log_amount_received"]
    ).where(X["is_currency_same"].eq(1), 0).astype("float32")
    X["is_round_10"] = np.isclose(np.mod(paid, 10), 0, atol=1e-8).astype("int8")
    X["is_round_100"] = np.isclose(np.mod(paid, 100), 0, atol=1e-8).astype("int8")
    X["is_round_1000"] = np.isclose(np.mod(paid, 1_000), 0, atol=1e-8).astype("int8")

    return X

def _composite_key(X: pd.DataFrame, columns) -> pd.Series:
    """Stable compact key for bank-scoped account identities and interactions."""
    return pd.util.hash_pandas_object(
        X.loc[:, list(columns)],
        index=False,
    ).astype("uint64")

def _strict_prior_count(entity: pd.Series, time_group: pd.Series) -> pd.Series:
    """Number of entity rows from strictly earlier timestamp groups."""
    row_count_before = entity.groupby(entity, sort=False).cumcount()
    same_time_before = entity.groupby([entity, time_group], sort=False).cumcount()
    return (row_count_before - same_time_before).astype("int32")

def _strict_prior_cumsum(
    values: pd.Series,
    entity: pd.Series,
    time_group: pd.Series,
) -> pd.Series:
    """Cumulative sum over strictly earlier timestamp groups."""
    # Shift whole batch totals instead of subtracting two large floating sums:
    # (1 + 1e20) - 1e20 loses the historical 1 in float64.
    batches = pd.DataFrame({"entity": entity, "time": time_group, "value": values})
    totals = batches.groupby(["entity", "time"], sort=False)["value"].sum()
    cumulative = totals.groupby(level="entity", sort=False).cumsum()
    prior = cumulative.groupby(level="entity", sort=False).shift(fill_value=0)
    keys = pd.MultiIndex.from_arrays([entity, time_group], names=["entity", "time"])
    return pd.Series(prior.reindex(keys).to_numpy(), index=values.index)

def _strict_previous_timestamp(
    timestamp: pd.Series,
    entity: pd.Series,
) -> pd.Series:
    """Previous distinct timestamp; all rows in one timestamp batch share the same history."""
    previous_row_timestamp = timestamp.groupby(entity, sort=False).shift()
    previous_strict = previous_row_timestamp.where(previous_row_timestamp < timestamp)
    return previous_strict.groupby(entity, sort=False).ffill()

def historical_features(
    X: pd.DataFrame,
    timestamp: pd.Series,
) -> pd.DataFrame:
    X = X.reset_index(drop=True).copy()
    timestamp = pd.Series(timestamp).reset_index(drop=True)

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

class OrderedTargetEncoder:
    """
    Timestamp-aware ordered target encoder for binary classification.

    Training rows:
        TE(x_i) uses only labels from rows with timestamp < timestamp_i.
        Rows sharing the same timestamp never use one another's targets.

    Validation/test rows:
        use statistics fitted on the complete sampled training period.

    Downsampling correction:
        optional sample weights are used inside the target statistics, so the
        encoded rates remain aligned with the original class distribution.
    """

    def __init__(
        self,
        categorical_features,
        interactions=None,
        smoothing=(20.0, 200.0),
        add_log_count=True,
        initial_prior=0.0,
    ):
        self.categorical_features = list(categorical_features)
        self.interactions = [tuple(cols) for cols in (interactions or [])]
        self.smoothing = tuple(float(alpha) for alpha in smoothing)
        self.add_log_count = bool(add_log_count)
        self.initial_prior = float(initial_prior)
        if not self.smoothing or any(not np.isfinite(a) or a <= 0 for a in self.smoothing):
            raise ValueError("Smoothing must be finite and positive")
        if not 0 <= self.initial_prior <= 1:
            raise ValueError("Prior must lie in [0,1]")

    @staticmethod
    def _feature_name(columns):
        return "__".join(
            str(col).replace(" ", "_").replace(".", "_")
            for col in columns
        )

    @staticmethod
    def _key(X, columns):
        return pd.util.hash_pandas_object(
            X.loc[:, list(columns)],
            index=False,
        ).astype("uint64")

    def _specs(self):
        specs = [(col,) for col in self.categorical_features] + self.interactions
        return list(dict.fromkeys(specs))

    @staticmethod
    def _validate_timestamp(timestamp, n_rows):
        timestamp = pd.Series(timestamp).reset_index(drop=True)
        if len(timestamp) != n_rows:
            raise ValueError("X and timestamp must have the same number of rows")
        if timestamp.isna().any() or not timestamp.is_monotonic_increasing:
            raise ValueError("timestamp must be sorted in ascending order")
        return timestamp

    @staticmethod
    def _weights(sample_weight, n_rows):
        if sample_weight is None:
            return np.ones(n_rows, dtype=np.float64)

        weight = np.asarray(sample_weight, dtype=np.float64)
        if len(weight) != n_rows:
            raise ValueError("sample_weight must have the same number of rows as X")
        if not np.isfinite(weight).all() or np.any(weight <= 0):
            raise ValueError("sample_weight must be strictly positive")
        return weight

    def _fit_maps(self, X, y_array, weight_array):
        self.specs_ = self._specs()
        weighted_target = y_array * weight_array
        self.global_prior_ = float(
            weighted_target.sum() / weight_array.sum()
        )
        self.stats_ = {}

        for spec in self.specs_:
            key = self._key(X, spec).to_numpy()
            stats_frame = pd.DataFrame(
                {
                    "key": key,
                    "weighted_target": weighted_target,
                    "weight": weight_array,
                }
            )
            self.stats_[spec] = (
                stats_frame
                .groupby("key", sort=False, observed=True)
                .agg(
                    sum=("weighted_target", "sum"),
                    count=("weight", "sum"),
                )
            )

    def fit(self, X, y, sample_weight=None):
        y_array = np.asarray(y, dtype=np.float64)
        if not len(y_array) or not np.isin(y_array, [0, 1]).all():
            raise ValueError("Expected nonempty binary targets")
        if len(X) != len(y_array):
            raise ValueError("X and y must have the same number of rows")

        weight_array = self._weights(sample_weight, len(X))
        self._fit_maps(X, y_array, weight_array)
        return self

    def fit_transform(self, X, y, timestamp, sample_weight=None):
        y_array = np.asarray(y, dtype=np.float64)
        if not len(y_array) or not np.isin(y_array, [0, 1]).all():
            raise ValueError("Expected nonempty binary targets")
        n_rows = len(X)

        if n_rows != len(y_array):
            raise ValueError("X and y must have the same number of rows")

        timestamp = self._validate_timestamp(timestamp, n_rows)
        weight_array = self._weights(sample_weight, n_rows)
        time_group = timestamp.ne(timestamp.shift()).cumsum().astype("int32")

        target = pd.Series(y_array, copy=False)
        weight = pd.Series(weight_array, copy=False)
        weighted_target = target * weight

        # Weighted global prior from strictly earlier timestamp groups.
        global_target_before = (
            weighted_target.cumsum()
            - weighted_target.groupby(time_group, sort=False).cumsum()
        ).to_numpy(dtype=np.float64)
        global_weight_before = (
            weight.cumsum()
            - weight.groupby(time_group, sort=False).cumsum()
        ).to_numpy(dtype=np.float64)

        global_prior_before = np.divide(
            global_target_before,
            global_weight_before,
            out=np.full(n_rows, self.initial_prior, dtype=np.float64),
            where=global_weight_before > 0,
        )

        encoded = (
            X.drop(columns=self.categorical_features, errors="ignore")
            .reset_index(drop=True)
            .copy()
        )
        self.specs_ = self._specs()

        for spec in self.specs_:
            key = self._key(X, spec).reset_index(drop=True)

            # Same-timestamp rows are removed as one batch, so no tie leakage is possible.
            category_target_before = (
                weighted_target.groupby(key, sort=False).cumsum()
                - weighted_target.groupby([key, time_group], sort=False).cumsum()
            ).to_numpy(dtype=np.float64)

            category_weight_before = (
                weight.groupby(key, sort=False).cumsum()
                - weight.groupby([key, time_group], sort=False).cumsum()
            ).to_numpy(dtype=np.float64)

            name = self._feature_name(spec)
            for alpha in self.smoothing:
                rate = (
                    category_target_before + alpha * global_prior_before
                ) / (
                    category_weight_before + alpha
                )
                encoded[f"{name}__ote_{alpha:g}"] = rate.astype("float32")

            if self.add_log_count:
                encoded[f"{name}__hist_log_count"] = np.log1p(
                    category_weight_before
                ).astype("float32")

        self._fit_maps(X, y_array, weight_array)
        return encoded

    def transform(self, X):
        if not hasattr(self, "stats_"):
            raise RuntimeError("Call fit(...) or fit_transform(...) before transform(...)")

        encoded = (
            X.drop(columns=self.categorical_features, errors="ignore")
            .reset_index(drop=True)
            .copy()
        )

        for spec in self.specs_:
            key = self._key(X, spec).reset_index(drop=True)
            stats = self.stats_[spec]

            category_sum = (
                key.map(stats["sum"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )
            category_count = (
                key.map(stats["count"])
                .fillna(0.0)
                .to_numpy(dtype=np.float64)
            )

            name = self._feature_name(spec)
            for alpha in self.smoothing:
                rate = (
                    category_sum + alpha * self.global_prior_
                ) / (
                    category_count + alpha
                )
                encoded[f"{name}__ote_{alpha:g}"] = rate.astype("float32")

            if self.add_log_count:
                encoded[f"{name}__hist_log_count"] = np.log1p(
                    category_count
                ).astype("float32")

        return encoded

def new_run(parent, group):
    """Allocate a unique output directory; never replace an existing experiment."""
    from datetime import datetime, timezone

    name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:12]
    path = Path(parent) / group / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def set_active(parent, filename, path):
    """Publish a completed run pointer atomically."""
    parent = Path(parent)
    temp = parent / (filename + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps({"path": str(Path(path).resolve().relative_to(parent.resolve()))}))
    temp.replace(parent / filename)


def resolve_training(run):
    run = Path(run)
    pointer = run / "active_training.json"
    path = run / json.loads(pointer.read_text())["path"] if pointer.exists() else run
    if not (path / "champion.joblib").exists():
        raise FileNotFoundError("Run notebook 03 first")
    return path


def error_summary(frame, target="target", prediction="prediction"):
    """Counts and operational metrics for a group, including one-class groups."""
    y = frame[target].astype(bool)
    pred = frame[prediction].astype(bool)
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    return pd.Series({
        "rows": len(frame),
        "positives": int(y.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "alerts": int(pred.sum()),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "alert_rate": pred.mean(),
    })


def rolling_windows(timestamp, n_folds=3, initial_days=3, validation_days=1, evaluation_days=1):
    """Build expanding training and fixed-duration validation/evaluation windows."""
    from datetime import timedelta

    t = pd.Series(timestamp).reset_index(drop=True)
    if t.empty or t.isna().any() or not t.is_monotonic_increasing:
        raise ValueError("Expected nonempty sorted timestamps")
    if min(n_folds, initial_days, validation_days, evaluation_days) <= 0:
        raise ValueError("Window sizes must be positive")

    result = []
    for fold in range(n_folds):
        train_end = t.iloc[0].floor("D").to_pydatetime() + timedelta(
            days=initial_days + fold * evaluation_days
        )
        val_end = train_end + timedelta(days=validation_days)
        eval_end = val_end + timedelta(days=evaluation_days)
        if eval_end > t.iloc[-1]:
            raise ValueError("Insufficient development period for complete backtest windows")
        edges = [0] + [int(t.searchsorted(value, side="left")) for value in (train_end, val_end, eval_end)]
        if any(a >= b for a, b in zip(edges, edges[1:])):
            raise ValueError("Empty backtest partition")
        result.append(dict(zip(
            ("train", "val", "evaluation"),
            (slice(a, b) for a, b in zip(edges, edges[1:])),
        )))
    return result


def budget_metrics(y, scores, timestamps, fraction):
    """Evaluate a daily retrospective top-k alert queue."""
    if not 0 < fraction <= 1:
        raise ValueError("Expected alert fraction in (0,1]")
    frame = pd.DataFrame({
        "target": np.asarray(y),
        "score": np.asarray(scores),
        "day": pd.to_datetime(np.asarray(timestamps)).floor("D"),
    })
    frame["prediction"] = False
    for _, group in frame.groupby("day", sort=True):
        k = int(np.floor(len(group) * fraction))
        picked = group.sort_values("score", ascending=False, kind="stable").head(k).index
        frame.loc[picked, "prediction"] = True
    return error_summary(frame)
