"""Run the Menkveld-style extras for the H1/H2 presentation layer.

Generates outputs that sit alongside the main presentation 
tables/figures:

1. universe-wide daily dark-share series (figure + CSV),
2. pre/post Menkveld-style descriptive stats table,
3. stacked cluster-diagnostics summary across VIX/macro/earnings.
"""

from __future__ import annotations

import importlib


_extras = importlib.import_module("14_h1_h2_menkveld_extras")

run_menkveld_extras = _extras.run_menkveld_extras


def main() -> None:
    outputs = run_menkveld_extras()

    print("\nSaved H1/H2 Menkveld extras:")
    for key, value in outputs.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for path in value:
                print(f"    - {path}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
