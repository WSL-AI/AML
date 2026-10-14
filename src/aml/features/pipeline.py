from __future__ import annotations

import pandas as pd

from aml.data.schema import TIMESTAMP
from aml.features.basic import build_basic_features
from aml.features.currency import CurrencyConverter
from aml.features.historical import HistoricalFeatureBuilder


class FeaturePipeline:
    def __init__(self, windows: dict[str, int] | None = None, anchor_currency: str = "US Dollar"):
        self.windows = windows
        self.anchor_currency = anchor_currency

    def fit_transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        self.converter_ = CurrencyConverter(self.anchor_currency).fit(transactions)
        self.history_ = HistoricalFeatureBuilder(self.windows)
        enriched = self._enrich(transactions)
        return self._combine(enriched, self.history_.fit_transform(enriched, enriched[TIMESTAMP]))

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        history = self._fitted()
        enriched = self._enrich(transactions)
        return self._combine(enriched, history.transform(enriched, enriched[TIMESTAMP]))

    def transform_update(self, transactions: pd.DataFrame) -> pd.DataFrame:
        history = self._fitted()
        enriched = self._enrich(transactions)
        return self._combine(enriched, history.transform_update(enriched, enriched[TIMESTAMP]))

    def _fitted(self) -> HistoricalFeatureBuilder:
        if not hasattr(self, "history_"):
            raise RuntimeError("Call fit_transform first")
        return self.history_

    def _enrich(self, transactions: pd.DataFrame) -> pd.DataFrame:
        frame = transactions.reset_index(drop=True)
        return pd.concat([frame, self.converter_.transform(frame)], axis=1)

    @staticmethod
    def _combine(enriched: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
        return pd.concat([build_basic_features(enriched), history], axis=1)
