"""Stable 64-bit identities for accounts and account pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd


def account_key(bank: pd.Series, account: pd.Series) -> np.ndarray:
    """Hash a bank-scoped account; sender and receiver roles share one identity space."""
    frame = pd.DataFrame({"bank": bank.to_numpy(), "account": account.to_numpy()})
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)


def pair_key(sender: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    """Hash an ordered sender-receiver relationship."""
    frame = pd.DataFrame({"sender": sender, "receiver": receiver})
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
