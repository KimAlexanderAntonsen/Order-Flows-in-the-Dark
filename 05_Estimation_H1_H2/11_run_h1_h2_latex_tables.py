"""Run the LaTeX export layer for the H1/H2 presentation tables."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ESTIMATION_DIR = Path(__file__).resolve().parent
if str(ESTIMATION_DIR) not in sys.path:
    sys.path.insert(0, str(ESTIMATION_DIR))

_latex = importlib.import_module("10_h1_h2_latex_tables")


def main() -> None:
    """Create the LaTeX table snippets."""

    paths = _latex.build_latex_tables()
    print("Created LaTeX tables:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()

