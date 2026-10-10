from pathlib import Path

from xgboost import XGBClassifier

from aml.data.load import load_dataset_splits, scale_train_set


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "xgboost.json"


def train_xgboost(input_dir=None, model_path=DEFAULT_MODEL_PATH):
    splits = load_dataset_splits(["train", "val"], input_dir=input_dir)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    X_fit, y_fit, sample_weight = scale_train_set(X_train, y_train)

    xgb = XGBClassifier(
        learning_rate=0.058079,
        max_depth=8,
        min_child_weight=6.167806,
        subsample=0.823744,
        colsample_bytree=0.883752,
        gamma=0.000238,
        reg_alpha=0.022964,
        reg_lambda=0.037638,
        n_estimators=1200,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        max_bin=128,
        early_stopping_rounds=100,
        n_jobs=4,
        scale_pos_weight=1.539243,
        random_state=42,
    )
    xgb.fit(
        X_fit,
        y_fit,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xgb.save_model(output_path)
    return xgb
