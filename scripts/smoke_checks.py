#!/usr/bin/env python3
"""Model-free smoke checks for code added after the 17-test pytest suite.

Covers the citation scorer, the agent's iteration-cap handling, and the
FastAPI service (with a fake pipeline, so no LLM is loaded). Kept outside
pytest so the suite stays at 17 tests; CI runs this after pytest.

Usage:
    python scripts/smoke_checks.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

from src import api  # noqa: E402
from src.agent import TRUNCATED_NOTICE, _final_answer  # noqa: E402
from src.eval import extract_cited_pages, score_citations  # noqa: E402
from src.rag import RAGResult, Retrieved  # noqa: E402


def check_citation_scoring() -> None:
    assert extract_cited_pages("Yes (p. 9).") == [9]
    assert extract_cited_pages("See (p. 9, p. 10) and (p. 3).") == [9, 10, 3]
    assert extract_cited_pages("no citation here") == []

    ok = score_citations("A (p. 9).", retrieved_pages=[9, 10], relevant_pages=[10])
    assert ok["cited"] and ok["accurate"] and not ok["gold"]

    invented = score_citations("A (p. 42).", retrieved_pages=[9], relevant_pages=[9])
    assert invented["cited"] and not invented["accurate"] and not invented["gold"]

    none = score_citations("A.", retrieved_pages=[9], relevant_pages=[9])
    assert not none["cited"] and not none["accurate"]


def check_agent_cap_handling() -> None:
    def msg(blocks, stop_reason):
        return SimpleNamespace(content=blocks, stop_reason=stop_reason)

    text = SimpleNamespace(type="text", text="Answer (p. 9).")
    tool = SimpleNamespace(type="tool_use", name="search_document", input={})

    assert _final_answer(msg([text], "end_turn")) == ("Answer (p. 9).", False)
    # Cap hit while Claude was still calling tools: no text, must not be silent.
    assert _final_answer(msg([tool], "tool_use")) == (TRUNCATED_NOTICE, True)
    assert _final_answer(None) == (TRUNCATED_NOTICE, True)


class _FakePipeline:
    def answer(self, query, k=4, mode="few_shot", max_new_tokens=256):
        ctx = [Retrieved(id=0, page=9, text="Transport rules.", score=0.8)]
        return RAGResult(
            answer="By road and rail (p. 9).",
            contexts=ctx,
            retrieval_latency=0.01,
            generation_latency=0.5,
            mode=mode,
        )


def check_api() -> None:
    api.app.dependency_overrides[api.get_pipeline] = lambda: _FakePipeline()
    client = TestClient(api.app)

    assert client.get("/health").json() == {"status": "ok"}

    r = client.post("/ask", json={"question": "How are materials transported?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"].endswith("(p. 9).")
    assert body["sources"][0]["page"] == 9
    assert body["backend"] == "local"

    assert client.post("/ask", json={"question": "x", "k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "x", "mode": "nope"}).status_code == 422
    assert client.post("/ask", json={"question": ""}).status_code == 422

    api.app.dependency_overrides.clear()


def main() -> None:
    for check in (check_citation_scoring, check_agent_cap_handling, check_api):
        check()
        print(f"ok  {check.__name__}")
    print("All smoke checks passed.")


if __name__ == "__main__":
    main()
