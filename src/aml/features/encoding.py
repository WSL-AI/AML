import numpy as np
import pandas as pd

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
