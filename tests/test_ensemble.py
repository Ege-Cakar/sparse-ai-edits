import numpy as np
import pandas as pd

from sparse_edits.ensemble import ensemble


def test_ensemble_standardizes_each_detector_on_the_reference_rows():
    rng = np.random.default_rng(0)
    a = rng.normal(size=100)
    scores = pd.DataFrame({"a": a, "b": 3 * a + 7}, index=np.arange(100, 200))   # b is a rescaled copy of a
    ref = np.arange(100, 150)
    s = ensemble(scores, ref)
    expected = (a - a[:50].mean()) / a[:50].std(ddof=1)
    assert np.allclose(s.to_numpy(), expected)
    assert abs(s.loc[ref].mean()) < 1e-12 and np.isclose(s.loc[ref].std(), 1)
