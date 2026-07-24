#!/usr/bin/env python3
"""CI gate: fail the build if retrieval recall@1 regresses.

Reads results/eval_metrics.json (written by `python -m src.eval`) and exits
non-zero if recall@1 drops below a threshold. The threshold sits a small
margin below the measured baseline (~0.83 on the committed index) to absorb
minor embedding-library version drift without being a rubber stamp.

Usage:
    python -m src.eval                        # writes results/eval_metrics.json
    python scripts/check_eval_regression.py   # exits 1 if recall@1 regressed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_METRICS_PATH = "results/eval_metrics.json"
DEFAULT_THRESHOLD = 0.75


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Gate CI on retrieval recall@1 regression."
    )
    p.add_argument("--metrics", default=DEFAULT_METRICS_PATH)
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sys.exit(check(args.metrics, args.threshold))


if __name__ == "__main__":
    main()
