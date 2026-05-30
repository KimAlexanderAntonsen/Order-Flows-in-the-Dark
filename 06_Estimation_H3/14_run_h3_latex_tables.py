"""Run the H3 LaTeX table export layer."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


H3_DIR = Path(__file__).resolve().parent
if str(H3_DIR) not in sys.path:
    sys.path.insert(0, str(H3_DIR))

latex_tables = importlib.import_module("13_h3_latex_tables")


def main() -> None:
    """Generate the current set of H3 LaTeX tables."""

    paths = latex_tables.build_all_tables()
    print("Saved H3 LaTeX tables:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
