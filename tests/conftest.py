import csv

import numpy as np
import pandas as pd

from aml.data.schema import TIMESTAMP_FORMAT

EUR_TO_USD = 1.1


def make_transactions(days=12, rows=3_000, accounts=40, seed=0, start="2025-01-01"):
    """Synthetic IBM-like transactions with timestamp ties, repeat pairs and two currencies."""
    rng = np.random.default_rng(seed)
    minutes = np.sort(rng.integers(0, days * 24 * 60, size=rows))
    banks = np.array(["001", "002", "003"])
    sender = rng.integers(0, accounts, size=rows)
    # Most payments go to a few favourite counterparties, so pairs repeat.
    receiver = np.where(rng.random(rows) < 0.6, (sender * 7 + rng.integers(0, 3, rows)) % accounts,
                        rng.integers(0, accounts, size=rows))
    paid = np.round(rng.lognormal(6, 2, size=rows), 2)
    pay_eur = rng.random(rows) < 0.3
    receive_eur = np.where(rng.random(rows) < 0.8, pay_eur, ~pay_eur)
    rate = np.where(pay_eur, EUR_TO_USD, 1.0) / np.where(receive_eur, EUR_TO_USD, 1.0)
    return pd.DataFrame({
        "Timestamp": pd.Timestamp(start) + pd.to_timedelta(minutes, unit="min"),
        "From Bank": banks[sender % 3],
        "Account": [f"{i:06d}" for i in sender],
        "To Bank": banks[receiver % 3],
        "Account.1": [f"{i:06d}" for i in receiver],
        "Amount Received": np.round(paid * rate, 2),
        "Receiving Currency": np.where(receive_eur, "Euro", "US Dollar"),
        "Amount Paid": paid,
        "Payment Currency": np.where(pay_eur, "Euro", "US Dollar"),
        "Payment Format": rng.choice(["ACH", "Cheque", "Wire"], size=rows),
        "Is Laundering": (rng.random(rows) < 0.05).astype(int),
    })


def write_ibm_csv(df: pd.DataFrame, path):
    """Write the IBM layout: slash timestamps and two columns both named Account."""
    columns = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Amount Received",
               "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format", "Is Laundering"]
    frame = df[columns].copy()
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"]).dt.strftime(TIMESTAMP_FORMAT)
    header = [name if name != "Account.1" else "Account" for name in columns]
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(frame.itertuples(index=False))
    return path
