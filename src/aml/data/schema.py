TARGET = "Is Laundering"
TIMESTAMP = "Timestamp"

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

MODEL_EXCLUDED_COLUMNS = [
    TIMESTAMP,
    TARGET,
]

RAW_DTYPES = {
    "From Bank": "string",
    "Account": "string",
    "To Bank": "string",
    "Account.1": "string",
    "Receiving Currency": "string",
    "Payment Currency": "string",
    "Payment Format": "string",
    "Is Laundering": "int8",
}