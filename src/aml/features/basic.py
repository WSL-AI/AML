import numpy as np
import pandas as pd

from aml.data.schema import CATEGORICAL


def build_basic_features(df: pd.DataFrame) -> pd.DataFrame:
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


basic_features = build_basic_features