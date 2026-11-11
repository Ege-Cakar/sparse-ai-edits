import numpy as np
import pandas as pd
import torch

from sparse_edits.thirdparty import assign_nulls, cached


def test_cached_scores_each_distinct_text_once_and_resumes(tmp_path):
    calls = []
    def score(batch):
        calls.extend(batch)
        return torch.tensor([float(len(t)) for t in batch])
    assert cached(score, ["aaa", "a", "aaa", "bb"], 2, tmp_path, chunk=2) == [3, 1, 3, 2]
    assert sorted(calls) == ["a", "aaa", "bb"]
    assert cached(score, ["bb", "cccc"], 2, tmp_path) == [2, 4] and calls[-1] == "cccc" and len(calls) == 4


def test_assign_nulls_shares_folds_by_base_and_pools_small_domains():
    df = pd.DataFrame(dict(dataset="d", domain=["a"] * 3 + ["b"] * 2 + ["c"], xgroup=["x"] * 4 + ["", ""],
                           base=["1", "1", "2", "3", "4", "5"], null_domain=[True, False, True, True, False, False]))
    out = assign_nulls(df, 0, min_null_docs=2)
    assert out.fold.iat[0] == out.fold.iat[1] and set(out.fold.iloc[:4]) <= {0, 1} and (out.fold.iloc[4:] == -1).all()
    assert out.scope_domain.tolist() == ["d/a"] * 3 + ["d/small_domains"] * 3
