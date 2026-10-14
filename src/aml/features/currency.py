from __future__ import annotations

import numpy as np
import pandas as pd

from aml.data.schema import PAID_USD, RECEIVED_USD


class CurrencyConverter:
    def __init__(self, anchor: str = "US Dollar"):
        self.anchor = anchor

    def fit(self, transactions: pd.DataFrame) -> "CurrencyConverter":
        paid = transactions["Amount Paid"].to_numpy(dtype=np.float64)
        received = transactions["Amount Received"].to_numpy(dtype=np.float64)
        pay = transactions["Payment Currency"].astype(str)
        receive = transactions["Receiving Currency"].astype(str)
        currencies = sorted(set(pay) | set(receive))
        if self.anchor not in currencies:
            raise ValueError(f"Anchor currency {self.anchor!r} is absent from the data")

        cross = (pay != receive).to_numpy() & (paid > 0) & (received > 0)
        observed = pd.DataFrame({
            "pay": pay[cross].to_numpy(),
            "receive": receive[cross].to_numpy(),
            "log_ratio": np.log(received[cross] / paid[cross]),
        })
        pairs = observed.groupby(["pay", "receive"])["log_ratio"].agg(["median", "size"]).reset_index()

        unlinked = set(currencies) - _linked(self.anchor, pairs)
        if unlinked:
            raise ValueError(f"Cannot infer exchange rates for: {sorted(unlinked)}")

        free = [currency for currency in currencies if currency != self.anchor]
        position = {currency: i for i, currency in enumerate(free)}
        design = np.zeros((len(pairs), len(free)))
        for row, (pay_currency, receive_currency) in enumerate(zip(pairs["pay"], pairs["receive"])):
            if pay_currency in position:
                design[row, position[pay_currency]] += 1.0
            if receive_currency in position:
                design[row, position[receive_currency]] -= 1.0
        weight = np.sqrt(pairs["size"].to_numpy(dtype=np.float64))
        solution = np.linalg.lstsq(design * weight[:, None], pairs["median"].to_numpy() * weight,
                                   rcond=None)[0] if free else np.empty(0)

        self.log_rates_ = pd.Series({self.anchor: 0.0, **dict(zip(free, solution))}).sort_index()
        residual = pairs["median"].to_numpy() - design @ solution if free else np.empty(0)
        self.max_abs_log_residual_ = float(np.abs(residual).max()) if len(residual) else 0.0
        return self

    def transform(self, transactions: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "log_rates_"):
            raise RuntimeError("Call fit before transform")
        result = {}
        for amount, currency, name in (("Amount Paid", "Payment Currency", PAID_USD),
                                       ("Amount Received", "Receiving Currency", RECEIVED_USD)):
            log_rate = transactions[currency].astype(str).map(self.log_rates_)
            if log_rate.isna().any():
                unknown = sorted(transactions.loc[log_rate.isna(), currency].astype(str).unique())
                raise ValueError(f"No exchange rate for currencies: {unknown}")
            result[name] = transactions[amount].to_numpy(dtype=np.float64) * np.exp(log_rate.to_numpy())
        return pd.DataFrame(result, index=transactions.index)

    def report(self) -> dict:
        return {
            "anchor": self.anchor,
            "log_rates": {currency: float(rate) for currency, rate in self.log_rates_.items()},
            "max_abs_log_residual": self.max_abs_log_residual_,
        }


def _linked(anchor: str, pairs: pd.DataFrame) -> set[str]:
    neighbours: dict[str, set[str]] = {}
    for left, right in zip(pairs["pay"], pairs["receive"]):
        neighbours.setdefault(left, set()).add(right)
        neighbours.setdefault(right, set()).add(left)
    linked, frontier = {anchor}, [anchor]
    while frontier:
        for neighbour in neighbours.get(frontier.pop(), ()):
            if neighbour not in linked:
                linked.add(neighbour)
                frontier.append(neighbour)
    return linked
