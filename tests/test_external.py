import numpy as np
import pandas as pd

from sparse_edits.external import doc_tests, score_all


def test_doc_tests_are_per_source_conformal_and_skip_calib():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"source": ["a"] * 6 + ["b"] * 3,
                       "split": ["calib"] * 4 + ["test"] * 2 + ["calib", "test", "test"],
                       "editor": "none", "score": [0, 1, 2, 3, 10, -1, 5, 9, 0]})
    t = doc_tests(df, 0.25, rng)
    assert len(t) == 4 and (t.split == "test").all()
    assert 0 < t.p.iloc[0] <= 1 / 5 and t.reject.iloc[0]   # above every calib score: randomized p = U / (4 + 1)


def test_score_all_resumes_from_saved_chunks(tmp_path):
    calls = []
    def score(batch):
        calls.append(len(batch))
        import torch
        return torch.tensor([float(len(t)) for t in batch])
    texts = ["aaa", "a", "aa", "aaaa", "b"]
    first = score_all(score, texts, 2, tmp_path, chunk=2)
    n = len(calls)
    again = score_all(score, texts, 2, tmp_path, chunk=2)
    assert first.tolist() == [3, 1, 2, 4, 1] and again.tolist() == first.tolist() and len(calls) == n
