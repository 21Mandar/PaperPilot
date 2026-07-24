"""Unit tests for src.rag.

Pure-function tests need no model. The retrieval test loads the small
sentence-transformer embedder (~80MB) and searches the FAISS index already
committed under index/ — it does NOT load the local Qwen generation model.
"""

import pytest

from src.rag import (
    FEW_SHOT_EXAMPLES,
    SYSTEM_PROMPT,
    Retrieved,
    Retriever,
    build_messages,
    format_context,
)


def test_format_context_renders_page_tags():
    contexts = [
        Retrieved(id=0, page=3, text="stacks are LIFO", score=0.9),
        Retrieved(id=1, page=5, text="queues are FIFO", score=0.8),
    ]
    rendered = format_context(contexts)
    assert "[p. 3]" in rendered
    assert "[p. 5]" in rendered
    assert "stacks are LIFO" in rendered


def test_build_messages_zero_shot_has_no_examples():
    contexts = [Retrieved(id=0, page=1, text="hello", score=0.5)]
    messages = build_messages("what?", contexts, mode="zero_shot")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert len(messages) == 2  # system + user, no few-shot turns
    assert messages[-1]["role"] == "user"
    assert "what?" in messages[-1]["content"]


def test_build_messages_few_shot_includes_examples():
    contexts = [Retrieved(id=0, page=1, text="hello", score=0.5)]
    messages = build_messages("what?", contexts, mode="few_shot")
    assert len(messages) == 2 + len(FEW_SHOT_EXAMPLES)


def test_build_messages_rejects_unknown_mode():
    with pytest.raises(ValueError):
        build_messages("q", [], mode="bogus")


def test_retriever_finds_relevant_pages_for_known_question():
    # Uses the FAISS index + chunks committed under index/ (built from
    # data/sample.pdf) — no rebuild needed. Question + expected pages are
    # taken straight from the hand-labeled data/eval.json.
    retriever = Retriever("index")
    results, latency = retriever.search(
        "How are radioactive materials transported?", k=5
    )
    assert results
    assert latency >= 0
    pages = {r.page for r in results}
    assert pages & {9, 10}
