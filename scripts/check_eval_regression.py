#!/usr/bin/env python3
"""CI gate: fail the build if retrieval or citation quality regresses.

Two checks:

1. Retrieval — reads results/eval_metrics.json (written by `python -m src.eval`)
   and fails if recall@1 drops below a threshold. The threshold sits a small
   margin below the measured baseline (~0.83 on the committed index) to absorb
   minor embedding-library version drift without being a rubber stamp.

2. Citations — reads the committed results/prompt_eval.json (written by
   `python -m src.eval --with-generation`, which needs the local LLM and so is
   not run in CI) and fails if the few-shot citation accuracy recorded there
   falls below a threshold. This keeps the committed record from being
   overwritten with a worse run, or from being edited, without CI noticing.
   Skipped (not failed) while no such record has been committed.

Usage:
    python -m src.eval                        # writes results/eval_metrics.json
    python scripts/check_eval_regression.py   # exits 1 if either regressed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_METRICS_PATH = "results/eval_metrics.json"
DEFAULT_THRESHOLD = 0.75
DEFAULT_PROMPT_METRICS_PATH = "results/prompt_eval.json"
DEFAULT_PROMPT_THRESHOLD = 0.9


def check(metrics_path: str | Path, threshold: float) -> int:
    path = Path(metrics_path)
    if not path.exists():
        print(f"No metrics file at {path} — run `python -m src.eval` first.")
        return 1

    payload = json.loads(path.read_text())
    recall_at_k = payload.get("retrieval", {}).get("recall_at_k", {})
    # Keys round-trip through JSON as strings even though src.eval builds the
    # dict with int keys.
    recall_at_1 = recall_at_k.get("1")

    if recall_at_1 is None:
        print(f"No recall@1 found in {path}. Available keys: {list(recall_at_k)}")
        return 1

    print(f"recall@1 = {recall_at_1:.3f}  (threshold: {threshold:.3f})")
    if recall_at_1 < threshold:
        print(
            f"FAIL: retrieval recall@1 ({recall_at_1:.3f}) dropped below the "
            f"{threshold:.3f} threshold."
        )
        return 1

    print("OK: retrieval recall@1 within threshold.")
    return 0


def check_citations(prompt_metrics_path: str | Path, threshold: float) -> int:
    path = Path(prompt_metrics_path)
    if not path.exists():
        # Nothing committed to compare against yet; not a regression.
        print(
            f"SKIPPED: no citation record at {path}. Run "
            "`python -m src.eval --with-generation` and commit the file to "
            "enable this gate."
        )
        return 0

    summary = json.loads(path.read_text()).get("summary", {})
    few_shot = summary.get("few_shot", {}).get("citation_accuracy")
    zero_shot = summary.get("zero_shot", {}).get("citation_accuracy")
    if few_shot is None or zero_shot is None:
        print(f"No citation_accuracy for both modes in {path}. Found: {list(summary)}")
        return 1

    print(
        f"citation accuracy: zero-shot {zero_shot:.3f} -> few-shot {few_shot:.3f}  "
        f"(few-shot threshold: {threshold:.3f})"
    )
    if few_shot < threshold:
        print(
            f"FAIL: few-shot citation accuracy ({few_shot:.3f}) is below the "
            f"{threshold:.3f} threshold."
        )
        return 1

    print("OK: few-shot citation accuracy within threshold.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gate CI on retrieval recall@1 and citation-accuracy regression."
    )
    p.add_argument("--metrics", default=DEFAULT_METRICS_PATH)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--prompt-metrics", default=DEFAULT_PROMPT_METRICS_PATH)
    p.add_argument("--prompt-threshold", type=float, default=DEFAULT_PROMPT_THRESHOLD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_rc = check(args.metrics, args.threshold)
    citation_rc = check_citations(args.prompt_metrics, args.prompt_threshold)
    sys.exit(retrieval_rc or citation_rc)


if __name__ == "__main__":
    main()
