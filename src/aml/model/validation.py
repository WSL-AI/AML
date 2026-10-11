def lightgbm_validation(X, y):
    """Use the preferred API on new LightGBM and retain compatibility with 4.x"""
    import inspect
    from lightgbm import LGBMClassifier
    if "eval_X" in inspect.signature(LGBMClassifier.fit).parameters:
        return {"eval_X": X, "eval_y": y}
    return {"eval_set": [(X, y)]}
