import pandas as pd

from sparse_edits.generate import body, pilot_docs, rejection


def test_body_raises_each_apis_output_cap_and_keeps_the_other_params():
    luna = dict(model="m", product="ChatGPT", params={"reasoning_effort": "none", "max_completion_tokens": 1024})
    haiku = dict(model="h", product="Claude", api="anthropic", params={"max_tokens": 1024, "thinking": {"type": "x"}})
    assert body(luna, "hi", 30000) | {"messages": None} == dict(model="m", reasoning_effort="none",
                                                                max_completion_tokens=30000, messages=None)
    b = body(haiku, "hi", 30000)
    assert b["max_tokens"] == 30000 and b["thinking"] == {"type": "x"} and b["messages"][0]["content"] == "hi"
    assert b["system"].startswith("You are Claude")


def test_rejection_of_cut_off_empty_and_multiline_replies():
    assert rejection("text", {"completion_tokens": 100}, 100) == "truncated"
    assert rejection("text", {"output_tokens": 99}, 100) is None
    assert rejection("  \n", {"output_tokens": 5}, 100) == "empty"
    assert rejection("a\nb", {"output_tokens": 5}, 100, one_line=True) == "not_one_line"
    assert rejection("a\nb", {"output_tokens": 5}, 100) is None


def test_pilot_takes_k_documents_per_source_in_a_fixed_order():
    docs = pd.DataFrame({"doc_id": range(40), "source": ["a", "b"] * 20})
    first = pilot_docs({"seed": 0}, docs, 3)
    assert docs[docs.doc_id.isin(first)].source.value_counts().to_dict() == {"a": 3, "b": 3}
    assert first == pilot_docs({"seed": 0}, docs, 3) and first < pilot_docs({"seed": 0}, docs, 5)
