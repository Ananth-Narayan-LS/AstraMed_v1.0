
"""Command-line entry point for AstraMed model comparison."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.comparison import generate_reports


if __name__ == "__main__":
    generate_reports()
