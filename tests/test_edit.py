import pandas as pd

from sparse_edits.edit import limited


def test_limit_keeps_split_shares_and_a_larger_limit_extends_the_same_order():
    split = ["train", "train", "val", "test"]   # shares 0.5, 0.25, 0.25
    sents = pd.DataFrame({"doc_id": [d for d in range(40) for _ in range(5)], "n_words": 3,
                          "split": [split[d % 4] for d in range(40) for _ in range(5)]})
    cfg = {"seed": 0, "edit": {"splits": ["train", "val", "test"]}}
    jobs = [(row, None, 0) for row in sents.itertuples()]
    first = limited(cfg, sents, jobs, 40)
    assert pd.Series([j[0].split for j in first]).value_counts().to_dict() == {"train": 20, "val": 10, "test": 10}
    assert {j[0].Index for j in first} < {j[0].Index for j in limited(cfg, sents, jobs, 80)}
    cfg["edit"]["splits"] = ["test"]
    assert {j[0].split for j in limited(cfg, sents, jobs, 40)} == {"test"}
