import joblib
import numpy as np
import pandas as pd

from aml.features.encoding import OrderedTargetEncoder


def test_encoding_excludes_own_timestamp_and_preserves_weights(tmp_path):
    X = pd.DataFrame({"kind": ["a"] * 4, "amount": [1., 2., 3., 4.]})
    t = pd.to_datetime(["2025-01-01"] * 2 + ["2025-01-02"] * 2)
    encoder = OrderedTargetEncoder(["kind"], smoothing=(2,))
    result = encoder.fit_transform(X, [0, 1, 0, 1], t, [9., 1., 9., 1.])
    np.testing.assert_allclose(result.kind__ote_2, [0, 0, .1, .1])
    other = OrderedTargetEncoder(["kind"], smoothing=(2,)).fit_transform(
        X, [0, 1, 1, 0], t, [9., 1., 9., 1.]
    )
    pd.testing.assert_frame_equal(result, other)
    unseen = pd.DataFrame({"kind": ["new"], "amount": [5.]})
    transformed = encoder.transform(unseen)
    np.testing.assert_allclose(transformed.kind__ote_2, [.1])
    path = tmp_path / "encoder.joblib"
    joblib.dump(encoder, path)
    loaded = joblib.load(path)
    assert type(loaded).__module__ == "aml.features.encoding"
    pd.testing.assert_frame_equal(loaded.transform(unseen), transformed)
