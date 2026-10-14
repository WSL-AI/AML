import numpy as np
import pandas as pd

from aml.data.schema import CATEGORICAL, PAID_USD, RECEIVED_USD, RECEIVER, SENDER, TIMESTAMP


def build_basic_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features; expects USD amounts added by ``CurrencyConverter``."""
    df = transactions.reset_index(drop=True)
    X = df[CATEGORICAL].copy()

    hour = df[TIMESTAMP].dt.hour
    X["hour"] = hour.astype("int8")
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")
    X["is_night"] = hour.between(0, 5).astype("int8")

    same_bank = df[SENDER[0]].eq(df[RECEIVER[0]])
    X["is_currency_same"] = df["Receiving Currency"].eq(df["Payment Currency"]).astype("int8")
    X["is_same_bank"] = same_bank.astype("int8")
    X["is_self_transfer"] = (same_bank & df[SENDER[1]].eq(df[RECEIVER[1]])).astype("int8")

    paid_usd = df[PAID_USD].clip(lower=0)
    received_usd = df[RECEIVED_USD].clip(lower=0)
    X["amount_paid_usd_log"] = np.log1p(paid_usd).astype("float32")
    X["amount_received_usd_log"] = np.log1p(received_usd).astype("float32")
    # Log ratio to the inferred exchange rate; zero for a payment at the market rate.
    priced = (paid_usd > 0) & (received_usd > 0)
    X["fx_log_deviation"] = np.log(
        received_usd.where(priced, 1.0) / paid_usd.where(priced, 1.0)
    ).astype("float32")

    # Roundness is a property of the amount as typed, i.e. in its own currency.
    paid = df["Amount Paid"].clip(lower=0)
    for base in (10, 100, 1_000):
        X[f"is_round_{base}"] = np.isclose(np.mod(paid, base), 0, atol=1e-8).astype("int8")
    return X
