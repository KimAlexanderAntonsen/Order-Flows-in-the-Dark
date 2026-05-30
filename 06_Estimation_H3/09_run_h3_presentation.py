"""Run the first presentation layer for H3."""

from __future__ import annotations

import importlib


_presentation = importlib.import_module("08_h3_presentation")

run_presentation_layer = _presentation.run_presentation_layer


def main() -> None:
    """Build the H3 presentation outputs and report what was saved."""

    outputs = run_presentation_layer()

    print("Saved H3 presentation outputs.\n")
    print("Tables")
    for path in outputs["tables"]:
        print(f"  - {path}")

    print("\nFigures")
    for path in outputs["figures"]:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
