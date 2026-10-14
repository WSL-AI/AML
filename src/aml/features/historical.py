from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aml.data.keys import account_key, pair_key
from aml.data.schema import PAID_USD, RECEIVED_USD, RECEIVER, SENDER

# Seconds reserved per entity in the sorted key: about 544 years of history.
SLOT = 1 << 34
DEFAULT_WINDOWS = {"1h": 3_600, "24h": 86_400}
REQUIRED_COLUMNS = (*SENDER, *RECEIVER, PAID_USD, RECEIVED_USD)


@dataclass(frozen=True)
class _Located:
    codes: np.ndarray   # entity code of each query, -1 for an entity the log has never seen
    start: np.ndarray   # first log position of the entity
    end: np.ndarray     # first log position at or after the query time
    times: np.ndarray   # query time in seconds since the builder origin
    known: np.ndarray


class _EventLog:
    """Immutable-by-convention event history; ``extended`` and ``trimmed`` return new logs."""

    def __init__(self, columns: tuple[str, ...] = (), track_times: bool = False):
        self.columns = tuple(columns)
        self.track_times = track_times
        self.vocabulary = pd.Index(np.empty(0, dtype=np.uint64))
        self.keys = np.empty(0, dtype=np.int64)
        self.values = {name: np.empty(0) for name in self.columns}
        # 1 for the first event of an (entity, timestamp) group: counts distinct activity times.
        self.first_at_time = np.empty(0, dtype=np.int64)
        self.archive = self._empty_archive(0)
        # Arrays derived from the immutable keys and values: codes, times and prefix sums.
        self._cache: dict = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    # -- construction ---------------------------------------------------------------------

    def extended(self, entities: np.ndarray, times: np.ndarray, values: dict) -> "_EventLog":
        """Return a log that also holds a batch strictly later than every logged event."""
        log = self._copy()
        codes = log.vocabulary.get_indexer(entities)
        if (codes < 0).any():
            log.vocabulary = log.vocabulary.append(pd.Index(pd.unique(entities[codes < 0])))
            if len(log.vocabulary) >= np.iinfo(np.int64).max // SLOT:
                raise OverflowError("Too many entities for the event key")
            codes = log.vocabulary.get_indexer(entities)
            padding = self._empty_archive(len(log.vocabulary) - len(log.archive["count"]))
            log.archive = {name: np.concatenate([array, padding[name]]) for name, array in log.archive.items()}

        batch_keys = codes.astype(np.int64) * SLOT + times
        order = np.argsort(batch_keys, kind="stable")
        batch_keys = batch_keys[order]
        first = np.ones(len(batch_keys), dtype=np.int64)
        first[1:] = batch_keys[1:] != batch_keys[:-1]

        positions = np.searchsorted(log.keys, batch_keys, side="right")
        log.keys = np.insert(log.keys, positions, batch_keys)
        log.first_at_time = np.insert(log.first_at_time, positions, first)
        log.values = {
            name: np.insert(log.values[name], positions, np.asarray(values[name], dtype=np.float64)[order])
            for name in self.columns
        }
        return log

    def trimmed(self, cutoff: int) -> "_EventLog":
        """Fold events at or before ``cutoff`` seconds into per-entity totals."""
        old = self._times() <= cutoff
        if not old.any():
            return self
        log = self._copy()
        codes = self._codes()[old]
        size = len(log.vocabulary)
        archive = dict(log.archive)
        archive["count"] = archive["count"] + np.bincount(codes, minlength=size)
        for name in self.columns:
            archive[name] = archive[name] + np.bincount(codes, self.values[name][old], minlength=size)
        if self.track_times:
            times = self._times()[old]
            archive["times"] = archive["times"] + np.bincount(
                codes, self.first_at_time[old], minlength=size
            ).astype(np.int64)
            # Events are sorted by (entity, time): first and last occurrence give min and max.
            entity, first_index = np.unique(codes, return_index=True)
            last_index = len(codes) - 1 - np.unique(codes[::-1], return_index=True)[1]
            archive["first"] = archive["first"].copy()
            unset = archive["first"][entity] < 0
            archive["first"][entity[unset]] = times[first_index[unset]]
            archive["last"] = archive["last"].copy()
            archive["last"][entity] = times[last_index]
        log.archive = archive
        log.keys = self.keys[~old]
        log.first_at_time = self.first_at_time[~old]
        log.values = {name: array[~old] for name, array in self.values.items()}
        return log

    # -- queries --------------------------------------------------------------------------

    def locate(self, entities: np.ndarray, times: np.ndarray) -> _Located:
        codes = self.vocabulary.get_indexer(entities).astype(np.int64)
        known = codes >= 0
        base = np.where(known, codes, 0) * SLOT
        start = np.where(known, np.searchsorted(self.keys, base, side="left"), 0)
        end = np.where(known, np.searchsorted(self.keys, base + times, side="left"), 0)
        return _Located(codes=codes, start=start, end=end, times=times, known=known)

    def count(self, at: _Located, window: int | None = None) -> np.ndarray:
        if window is not None:
            return at.end - self._window_start(at, window)
        return at.end - at.start + self._archived("count", at)

    def total(self, name: str, at: _Located, window: int | None = None) -> np.ndarray:
        if window is not None:
            return self._segment_sum(name, at, self._window_start(at, window))
        return self._segment_sum(name, at, at.start) + self._archived(name, at)

    def last_time(self, at: _Located) -> np.ndarray:
        """Most recent strictly earlier activity time, NaN if none."""
        logged = at.end > at.start
        previous = self._times()[np.maximum(at.end - 1, 0)] if len(self.keys) else np.zeros(len(at.end))
        archived = self._archived("last", at, missing=-1)
        return np.where(logged, previous, np.where(archived >= 0, archived, np.nan))

    def first_time(self, at: _Located) -> np.ndarray:
        """Earliest strictly earlier activity time, NaN if none."""
        archived = self._archived("first", at, missing=-1)
        logged = at.end > at.start
        first = self._times()[np.minimum(at.start, max(len(self.keys) - 1, 0))] \
            if len(self.keys) else np.zeros(len(at.end))
        return np.where(archived >= 0, archived, np.where(logged, first, np.nan))

    def distinct_times(self, at: _Located) -> np.ndarray:
        """Number of distinct strictly earlier activity timestamps."""
        logged = self._segment_sum("__first_at_time", at, at.start)
        return np.rint(logged).astype(np.int64) + self._archived("times", at)

    # -- internals ------------------------------------------------------------------------

    def _window_start(self, at: _Located, window: int) -> np.ndarray:
        # Events inside the window satisfy time > query time - window. A negative bound is
        # clipped to -1, which still sorts after every key of the previous entity.
        lower = np.maximum(at.times - window, -1)
        base = np.where(at.known, at.codes, 0) * SLOT
        start = np.searchsorted(self.keys, base + lower, side="right")
        return np.where(at.known, np.clip(start, at.start, at.end), 0)

    def _segment_sum(self, name: str, at: _Located, lower: np.ndarray) -> np.ndarray:
        """Sum of values at log positions [lower, end) inside each queried entity."""
        if not len(self.keys):
            return np.zeros(len(at.end))
        high, low = self._prefix(name)
        nonempty = at.end > lower
        upper = np.maximum(at.end - 1, 0)
        below = np.maximum(lower - 1, 0)
        inner = nonempty & (lower > at.start)
        # Close prefixes subtract exactly (Sterbenz); the low parts restore the rounding lost
        # when a small value was added to a large running total.
        result = (np.where(nonempty, high[upper], 0.0) - np.where(inner, high[below], 0.0)) \
            + (np.where(nonempty, low[upper], 0.0) - np.where(inner, low[below], 0.0))
        return np.maximum(result, 0.0)

    def _prefix(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-entity compensated prefix sums as (high, low) parts.

        Prefixes restart for every entity, and the rounding error of each running-sum step
        (TwoSum) is accumulated separately, so a window sum stays accurate even right after
        transactions many orders of magnitude larger.
        """
        key = ("prefix", name)
        if key not in self._cache:
            values = self.first_at_time if name == "__first_at_time" else self.values[name]
            values = pd.Series(values, dtype=np.float64)
            codes = self._codes()
            high = values.groupby(codes, sort=False).cumsum()
            previous = high.groupby(codes, sort=False).shift(fill_value=0.0)
            high, previous, values = high.to_numpy(), previous.to_numpy(), values.to_numpy()
            virtual = high - previous
            error = (previous - (high - virtual)) + (values - virtual)
            low = pd.Series(error).groupby(codes, sort=False).cumsum().to_numpy()
            self._cache[key] = (high, low)
        return self._cache[key]

    def _archived(self, name: str, at: _Located, missing=0) -> np.ndarray:
        array = self.archive[name]
        if not len(array):
            return np.full(len(at.codes), missing, dtype=array.dtype)
        return np.where(at.known, array[np.maximum(at.codes, 0)], missing)

    def _codes(self) -> np.ndarray:
        if "codes" not in self._cache:
            self._cache["codes"] = self.keys // SLOT
        return self._cache["codes"]

    def _times(self) -> np.ndarray:
        if "times" not in self._cache:
            self._cache["times"] = self.keys % SLOT
        return self._cache["times"]

    def _empty_archive(self, size: int) -> dict[str, np.ndarray]:
        archive = {"count": np.zeros(size, dtype=np.int64)}
        archive.update({name: np.zeros(size) for name in self.columns})
        if self.track_times:
            archive.update(
                times=np.zeros(size, dtype=np.int64),
                first=np.full(size, -1, dtype=np.int64),
                last=np.full(size, -1, dtype=np.int64),
            )
        return archive

    def _copy(self) -> "_EventLog":
        log = object.__new__(_EventLog)
        log.__dict__.update(self.__dict__)
        log._cache = {}
        return log


class HistoricalFeatureBuilder:
    """Stateful point-in-time features for chronologically arriving batches.

    ``fit_transform`` starts a new history, ``transform`` scores a later batch
    without remembering it, and ``transform_update`` scores it and appends it.
    Inside a batch, rows see earlier timestamps of the same batch, so feeding
    periods one after another equals one pass over the concatenated data.
    """

    def __init__(self, windows: dict[str, int] | None = None):
        windows = dict(DEFAULT_WINDOWS if windows is None else windows)
        if not windows or any(int(seconds) <= 0 for seconds in windows.values()):
            raise ValueError("Windows must be positive durations in seconds")
        self.windows = {name: int(seconds) for name, seconds in windows.items()}
        self.retention = max(self.windows.values())

    def fit(self, transactions: pd.DataFrame, timestamps) -> "HistoricalFeatureBuilder":
        self.fit_transform(transactions, timestamps)
        return self

    def fit_transform(self, transactions: pd.DataFrame, timestamps) -> pd.DataFrame:
        seconds = _seconds(timestamps)
        if not len(seconds):
            raise ValueError("Expected a nonempty batch")
        self.origin_ = int(seconds[0])
        self.last_time_ = -1
        self.outgoing_ = _EventLog(("amount", "interbank"), track_times=True)
        self.incoming_ = _EventLog(("amount",), track_times=True)
        self.pairs_ = _EventLog()
        self.sender_new_pairs_ = _EventLog()
        self.receiver_new_pairs_ = _EventLog()
        return self._step(transactions, seconds, update=True)

    def transform(self, transactions: pd.DataFrame, timestamps) -> pd.DataFrame:
        return self._step(transactions, self._checked_seconds(timestamps), update=False)

    def transform_update(self, transactions: pd.DataFrame, timestamps) -> pd.DataFrame:
        return self._step(transactions, self._checked_seconds(timestamps), update=True)

    @property
    def feature_names(self) -> list[str]:
        names = [
            "sender_prev_tx_log", "receiver_prev_tx_log", "pair_prev_tx_log", "is_new_pair",
            "sender_unique_receivers_log", "receiver_unique_senders_log",
            "sender_prior_interbank_ratio", "sender_minutes_since_prev_log",
            "receiver_minutes_since_prev_log", "sender_prior_avg_minutes_log",
            "amount_to_sender_prior_mean_log", "amount_to_receiver_prior_mean_log",
        ]
        for window in self.windows:
            names += [
                f"sender_tx_{window}_log", f"sender_amount_{window}_log",
                f"receiver_tx_{window}_log", f"receiver_amount_{window}_log",
                f"sender_new_receivers_{window}_log", f"receiver_new_senders_{window}_log",
                f"sender_inflow_{window}_log", f"sender_outflow_to_inflow_{window}_log",
            ]
        return names

    def _checked_seconds(self, timestamps) -> np.ndarray:
        if not hasattr(self, "origin_"):
            raise RuntimeError("Call fit or fit_transform first")
        seconds = _seconds(timestamps)
        if not len(seconds):
            raise ValueError("Expected a nonempty batch")
        if seconds[0] - self.origin_ <= self.last_time_:
            raise ValueError("New batch must be strictly after saved history; keep timestamp groups together")
        return seconds

    def _step(self, transactions: pd.DataFrame, seconds: np.ndarray, update: bool) -> pd.DataFrame:
        missing = set(REQUIRED_COLUMNS) - set(transactions.columns)
        if missing:
            raise ValueError(f"Missing history columns: {sorted(missing)}")
        if len(transactions) != len(seconds):
            raise ValueError("Transactions and timestamps must have the same number of rows")
        df = transactions.reset_index(drop=True)
        t = seconds - self.origin_

        sender = account_key(df[SENDER[0]], df[SENDER[1]])
        receiver = account_key(df[RECEIVER[0]], df[RECEIVER[1]])
        pair = pair_key(sender, receiver)
        paid = df[PAID_USD].clip(lower=0).to_numpy(dtype=np.float64)
        received = df[RECEIVED_USD].clip(lower=0).to_numpy(dtype=np.float64)
        interbank = df[SENDER[0]].ne(df[RECEIVER[0]]).to_numpy(dtype=np.float64)

        # A pair's first-ever transaction adds a new counterparty to both accounts.
        new_pair = (self.pairs_.vocabulary.get_indexer(pair) < 0) & ~pd.Series(pair).duplicated().to_numpy()

        outgoing = self.outgoing_.extended(sender, t, {"amount": paid, "interbank": interbank})
        incoming = self.incoming_.extended(receiver, t, {"amount": received})
        pairs = self.pairs_.extended(pair, t, {})
        sender_new = self.sender_new_pairs_.extended(sender[new_pair], t[new_pair], {})
        receiver_new = self.receiver_new_pairs_.extended(receiver[new_pair], t[new_pair], {})

        features = _features(
            self.windows, t, paid, received,
            out=(outgoing, outgoing.locate(sender, t)),
            inc=(incoming, incoming.locate(receiver, t)),
            inflow=(incoming, incoming.locate(sender, t)),
            pair=(pairs, pairs.locate(pair, t)),
            sender_new=(sender_new, sender_new.locate(sender, t)),
            receiver_new=(receiver_new, receiver_new.locate(receiver, t)),
        )

        if update:
            latest = int(t[-1])
            cutoff = latest - self.retention
            self.outgoing_ = outgoing.trimmed(cutoff)
            self.incoming_ = incoming.trimmed(cutoff)
            self.sender_new_pairs_ = sender_new.trimmed(cutoff)
            self.receiver_new_pairs_ = receiver_new.trimmed(cutoff)
            self.pairs_ = pairs.trimmed(latest)  # only all-time counts are needed for pairs
            self.last_time_ = latest
        return features[self.feature_names]


def _features(windows, t, paid, received, *, out, inc, inflow, pair, sender_new, receiver_new):
    out_log, out_at = out
    inc_log, inc_at = inc
    inflow_log, inflow_at = inflow
    f = {}

    sender_count = out_log.count(out_at)
    receiver_count = inc_log.count(inc_at)
    pair_count = pair[0].count(pair[1])
    f["sender_prev_tx_log"] = np.log1p(sender_count)
    f["receiver_prev_tx_log"] = np.log1p(receiver_count)
    f["pair_prev_tx_log"] = np.log1p(pair_count)
    f["is_new_pair"] = (pair_count == 0).astype("int8")
    f["sender_unique_receivers_log"] = np.log1p(sender_new[0].count(sender_new[1]))
    f["receiver_unique_senders_log"] = np.log1p(receiver_new[0].count(receiver_new[1]))
    f["sender_prior_interbank_ratio"] = _ratio(out_log.total("interbank", out_at), sender_count, fill=0.0)

    sender_last = out_log.last_time(out_at)
    f["sender_minutes_since_prev_log"] = _log_minutes(t - sender_last)
    f["receiver_minutes_since_prev_log"] = _log_minutes(t - inc_log.last_time(inc_at))
    # Mean gap between consecutive distinct activity times telescopes to (last - first) / gaps.
    gaps = out_log.distinct_times(out_at) - 1
    span = sender_last - out_log.first_time(out_at)
    f["sender_prior_avg_minutes_log"] = _log_minutes(np.where(gaps > 0, span, np.nan) / np.maximum(gaps, 1))

    f["amount_to_sender_prior_mean_log"] = _amount_to_mean(
        paid, out_log.total("amount", out_at), sender_count)
    f["amount_to_receiver_prior_mean_log"] = _amount_to_mean(
        received, inc_log.total("amount", inc_at), receiver_count)

    for name, seconds in windows.items():
        sender_amount = out_log.total("amount", out_at, seconds)
        inflow_amount = inflow_log.total("amount", inflow_at, seconds)
        f[f"sender_tx_{name}_log"] = np.log1p(out_log.count(out_at, seconds))
        f[f"sender_amount_{name}_log"] = np.log1p(sender_amount)
        f[f"receiver_tx_{name}_log"] = np.log1p(inc_log.count(inc_at, seconds))
        f[f"receiver_amount_{name}_log"] = np.log1p(inc_log.total("amount", inc_at, seconds))
        f[f"sender_new_receivers_{name}_log"] = np.log1p(sender_new[0].count(sender_new[1], seconds))
        f[f"receiver_new_senders_{name}_log"] = np.log1p(receiver_new[0].count(receiver_new[1], seconds))
        f[f"sender_inflow_{name}_log"] = np.log1p(inflow_amount)
        # Pass-through: money leaving an account shortly after arriving keeps this near log(2).
        f[f"sender_outflow_to_inflow_{name}_log"] = np.log1p(
            _ratio(sender_amount + paid, inflow_amount, fill=np.nan, upper=1e6)
        )

    return pd.DataFrame({
        name: values if values.dtype == np.int8 else np.nan_to_num(values, nan=-1.0).astype("float32")
        for name, values in f.items()
    })


def _ratio(numerator, denominator, fill, upper=None):
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), fill)
    return ratio if upper is None else np.minimum(ratio, upper)


def _log_minutes(seconds):
    return np.log1p(np.maximum(seconds, 0) / 60)


def _amount_to_mean(amount, total, count):
    mean = _ratio(total, count, fill=0.0)
    return np.log1p(_ratio(amount, mean, fill=0.0, upper=1e6))


def _seconds(timestamps) -> np.ndarray:
    timestamps = pd.to_datetime(pd.Series(timestamps), errors="raise").reset_index(drop=True)
    if timestamps.isna().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("timestamps must be sorted in ascending order without NaT")
    # Flooring to seconds can only merge timestamps, which hides information, never leaks it.
    return timestamps.to_numpy(dtype="datetime64[s]").astype(np.int64)


def build_historical_features(transactions: pd.DataFrame, timestamps, windows=None) -> pd.DataFrame:
    """One-shot historical features for a single chronological frame."""
    return HistoricalFeatureBuilder(windows).fit_transform(transactions, timestamps)
