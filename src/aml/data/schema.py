TARGET = "Is Laundering"
TIMESTAMP = "Timestamp"

# IBM CSV files format timestamps as "2022/09/01 00:20".
TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"

SENDER = ("From Bank", "Account")
RECEIVER = ("To Bank", "Account.1")

CATEGORICAL = [
    "From Bank",
    "Account",
    "To Bank",
    "Account.1",
    "Receiving Currency",
    "Payment Currency",
    "Payment Format",
]

AMOUNT = [
    "Amount Paid",
    "Amount Received",
]

RAW_COLUMNS = [
    TIMESTAMP,
    *CATEGORICAL,
    *AMOUNT,
    TARGET,
]

PAYMENT_FORMATS = frozenset({
    "ACH",
    "Bitcoin",
    "Cash",
    "Cheque",
    "Credit Card",
    "Reinvestment",
    "Wire",
})

# Amounts converted with exchange rates inferred from training transactions.
PAID_USD = "amount_paid_usd"
RECEIVED_USD = "amount_received_usd"
