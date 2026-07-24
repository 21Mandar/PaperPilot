"""Make the project root importable so `src` resolves regardless of how
pytest is invoked (bare `pytest` vs `python -m pytest`) or from where."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
