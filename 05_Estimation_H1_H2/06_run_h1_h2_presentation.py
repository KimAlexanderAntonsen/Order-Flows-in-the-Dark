"""Run the first presentation layer for H1/H2.

This script reuses the saved H1/H2 outputs to build:

1. Menkveld-inspired figures for the full sample,
2. compact summary tables for the regime comparison,
3. and VARX coefficient tables with simple significance markers.

The point is to move from estimation outputs to presentation-ready 
objects without burying the logic in a large notebook.
"""

from __future__ import annotations

import importlib


_presentation = importlib.import_module("05_h1_h2_presentation")

run_presentation_layer = _presentation.run_presentation_layer


def main() -> None:
    """Build the H1/H2 presentation outputs and report what was saved."""

    outputs = run_presentation_layer()

    print("Saved H1/H2 presentation outputs.\n")
    print("Tables")
    for path in outputs["tables"]:
        print(f"  - {path}")

    print("\nFigures")
    for path in outputs["figures"]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
