import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

def ranking_metrics(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    if len(y) != len(scores) or not len(y) or not np.isfinite(scores).all():
        raise ValueError("Invalid score vector")
    return {"AP": float(average_precision_score(y, scores)) if np.any(y == 1) else float("nan"),
            "ROC-AUC": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan")}

def error_summary(frame, target="target", prediction="prediction"):
    """Counts and operational metrics for a group, including one-class groups."""
    y = frame[target].astype(bool)
    pred = frame[prediction].astype(bool)
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    return pd.Series({
        "rows": len(frame),
        "positives": int(y.sum()),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "alerts": int(pred.sum()),
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "alert_rate": pred.mean(),
    })

def budget_metrics(y, scores, timestamps, fraction):
    """Evaluate a daily retrospective top-k alert queue."""
    if not 0 < fraction <= 1:
        raise ValueError("Expected alert fraction in (0,1]")
    frame = pd.DataFrame({
        "target": np.asarray(y),
        "score": np.asarray(scores),
        "day": pd.to_datetime(np.asarray(timestamps)).floor("D"),
    })
    frame["prediction"] = False
    for _, group in frame.groupby("day", sort=True):
        k = int(np.floor(len(group) * fraction))
        picked = group.sort_values("score", ascending=False, kind="stable").head(k).index
        frame.loc[picked, "prediction"] = True
    return error_summary(frame)
