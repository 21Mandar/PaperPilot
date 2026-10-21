"""PaperPilot — evaluation + prompt-comparison harness.

Two measurable things, both logged to ``results/``:

1. Retrieval quality: recall@k (a.k.a. hit-rate@k) over a small hand-labeled
   set in ``data/eval.json``, where each question is tagged with the page(s)
   that actually contain its answer. A query "hits" at k if any of its top-k
   retrieved chunks comes from a relevant page. Also reports retrieval latency.

2. Prompt comparison: zero-shot vs few-shot generation on the same questions,
   logging a citation rate (does the answer cite a page like "(p. 9)") and
   generation latency per mode. This is the iterate-and-measure loop that backs
   the "prompt engineering" claim.

Usage:
    python -m src.eval                       # retrieval metrics only (fast)
    python -m src.eval --with-generation     # also run zero/few-shot comparison
    python -m src.eval --ks 1 3 5 --k 5
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

import pandas as pd

from src.rag import (
    DEFAULT_INDEX_DIR,
    DEFAULT_LLM,
    Generator,
    Retriever,
    build_messages,
)

DEFAULT_EVAL_SET = "data/eval.json"
DEFAULT_RESULTS_DIR = "results"
PROMPT_EVAL_FILE = "prompt_eval.json"
DEFAULT_KS = [1, 3, 5]
CITATION_RE = re.compile(r"\(p\.\s*\d+", re.IGNORECASE)
CITED_PAGES_RE = re.compile(r"\(pp?\.\s*([^)]*)\)", re.IGNORECASE)


def extract_cited_pages(answer: str) -> list[int]:
    """Page numbers an answer cites, e.g. "(p. 9)" or "(p. 9, p. 10)"."""
    pages: list[int] = []
    for group in CITED_PAGES_RE.findall(answer):
        pages.extend(int(n) for n in re.findall(r"\d+", group))
    return pages


def score_citations(
    answer: str, retrieved_pages: list[int], relevant_pages: list[int]
) -> dict:
    """Score one answer's citations.

    ``cited``     the answer cites at least one page (citation rate).
    ``accurate``  it cites at least one page and every cited page was in the
                  retrieved context, i.e. no invented page numbers
                  (citation accuracy).
    ``gold``      at least one cited page is a labeled relevant page.
    """
    cited_pages = extract_cited_pages(answer)
    cited = bool(cited_pages)
    accurate = cited and all(p in set(retrieved_pages) for p in cited_pages)
    gold = any(p in set(relevant_pages) for p in cited_pages)
    return {
        "cited": cited,
        "accurate": accurate,
        "gold": gold,
        "cited_pages": cited_pages,
    }


def load_eval_set(path: str | Path) -> list[dict]:
    """Load the hand-labeled questions, skipping placeholder rows."""
    data = json.loads(Path(path).read_text())
    questions = data.get("questions", [])
    cleaned = [
        q for q in questions if not q["question"].startswith("REPLACE ME")
    ]
    if not cleaned:
        raise ValueError(
            f"No usable questions in {path}. Add real questions with "
            "relevant_pages before running eval."
        )
    return cleaned


def evaluate_retrieval(
    retriever: Retriever, eval_set: list[dict], ks: list[int]
) -> dict:
    """Compute recall@k for each k and retrieval-latency stats."""
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    latencies: list[float] = []
    per_question: list[dict] = []

    for item in eval_set:
        relevant = set(item["relevant_pages"])
        results, latency = retriever.search(item["question"], k=max_k)
        latencies.append(latency)
        retrieved_pages = [r.page for r in results]

        row = {"question": item["question"], "relevant_pages": sorted(relevant)}
        for k in ks:
            hit = any(p in relevant for p in retrieved_pages[:k])
            hits[k] += int(hit)
            row[f"hit@{k}"] = int(hit)
        row["top_pages"] = retrieved_pages
        per_question.append(row)

    n = len(eval_set)
    return {
        "num_questions": n,
        "recall_at_k": {k: hits[k] / n for k in ks},
        "retrieval_latency_ms": {
            "mean": 1000 * statistics.mean(latencies),
            "median": 1000 * statistics.median(latencies),
            "max": 1000 * max(latencies),
        },
        "per_question": per_question,
    }


def evaluate_prompts(
    retriever: Retriever,
    generator: Generator,
    eval_set: list[dict],
    k: int,
) -> dict:
    """Compare zero-shot vs few-shot: citation rate/accuracy + generation latency."""
    modes = ["zero_shot", "few_shot"]
    summary: dict = {}
    rows: list[dict] = []

    for mode in modes:
        cited = accurate = gold = 0
        gen_latencies: list[float] = []
        for item in eval_set:
            contexts, _ = retriever.search(item["question"], k=k)
            messages = build_messages(item["question"], contexts, mode=mode)
            answer, g_lat = generator.generate(messages)
            gen_latencies.append(g_lat)
            score = score_citations(
                answer, [c.page for c in contexts], item["relevant_pages"]
            )
            cited += int(score["cited"])
            accurate += int(score["accurate"])
            gold += int(score["gold"])
            rows.append(
                {
                    "mode": mode,
                    "question": item["question"],
                    "cited_page": score["cited"],
                    "citation_accurate": score["accurate"],
                    "cites_relevant_page": score["gold"],
                    "cited_pages": score["cited_pages"],
                    "gen_latency_s": round(g_lat, 2),
                    "answer": answer,
                }
            )
        n = len(eval_set)
        summary[mode] = {
            "citation_rate": cited / n,
            "citation_accuracy": accurate / n,
            "gold_page_citation_rate": gold / n,
            "gen_latency_s": {
                "mean": statistics.mean(gen_latencies),
                "median": statistics.median(gen_latencies),
            },
        }
    return {"summary": summary, "per_question": rows}


def save_results(
    out_dir: Path,
    retrieval: dict,
    prompts: dict | None,
    run_info: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"retrieval": retrieval}
    if prompts is not None:
        payload["prompts"] = prompts
    (out_dir / "eval_metrics.json").write_text(json.dumps(payload, indent=2))

    # Generation results live in their own committed file: the retrieval-only
    # run (CI) rewrites eval_metrics.json and would otherwise drop them.
    if prompts is not None:
        record = {**(run_info or {}), **prompts}
        (out_dir / PROMPT_EVAL_FILE).write_text(json.dumps(record, indent=2))


def print_report(retrieval: dict, prompts: dict | None) -> None:
    print(f"\n=== Retrieval ({retrieval['num_questions']} questions) ===")
    recall_df = pd.DataFrame(
        [{"k": k, "recall@k": v} for k, v in retrieval["recall_at_k"].items()]
    )
    print(recall_df.to_string(index=False))
    lat = retrieval["retrieval_latency_ms"]
    print(
        f"\nretrieval latency (ms): mean={lat['mean']:.1f}  "
        f"median={lat['median']:.1f}  max={lat['max']:.1f}"
    )

    if prompts is not None:
        print("\n=== Prompt comparison (zero-shot vs few-shot) ===")
        prompt_df = pd.DataFrame(
            [
                {
                    "mode": mode,
                    "citation_rate": s["citation_rate"],
                    "citation_accuracy": s["citation_accuracy"],
                    "gen_latency_s (mean)": round(s["gen_latency_s"]["mean"], 2),
                }
                for mode, s in prompts["summary"].items()
            ]
        )
        print(prompt_df.to_string(index=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PaperPilot retrieval + prompts.")
    p.add_argument("--eval-set", default=DEFAULT_EVAL_SET)
    p.add_argument("--index", default=DEFAULT_INDEX_DIR)
    p.add_argument("--model", default=DEFAULT_LLM, help="local LLM (generation only)")
    p.add_argument("--out", default=DEFAULT_RESULTS_DIR)
    p.add_argument("--k", type=int, default=4, help="k for the prompt comparison")
    p.add_argument("--ks", type=int, nargs="+", default=DEFAULT_KS,
                   help="k values for recall@k")
    p.add_argument("--with-generation", action="store_true",
                   help="also run the zero/few-shot prompt comparison (loads the LLM)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_set = load_eval_set(args.eval_set)
    retriever = Retriever(args.index)

    retrieval = evaluate_retrieval(retriever, eval_set, ks=sorted(args.ks))

    prompts = None
    if args.with_generation:
        generator = Generator(args.model)
        prompts = evaluate_prompts(retriever, generator, eval_set, k=args.k)

    run_info = {
        "model": args.model,
        "k": args.k,
        "num_questions": len(eval_set),
        "decoding": "greedy",
    }
    save_results(Path(args.out), retrieval, prompts, run_info=run_info)
    print_report(retrieval, prompts)
    print(f"\nSaved metrics to {args.out}/eval_metrics.json")


if __name__ == "__main__":
    main()
