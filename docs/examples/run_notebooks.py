#!/usr/bin/env python
"""
Execute all example Jupyter notebooks and report errors.

Usage:
    python run_notebooks.py              # run all notebooks
    python run_notebooks.py gic/01*      # run matching notebooks only

Notebooks are executed in-process with their directory as cwd so that
relative imports (e.g. ``from plot_helpers import ...``) resolve correctly.
Requires: jupyter, nbformat, nbclient  (pip install nbclient)
"""

import sys
import os
import time
import glob
import traceback
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

EXAMPLES_DIR = Path(__file__).resolve().parent


def find_notebooks(patterns=None):
    """Return sorted list of .ipynb paths under EXAMPLES_DIR."""
    if patterns:
        paths = []
        for pat in patterns:
            paths.extend(EXAMPLES_DIR.glob(pat))
    else:
        paths = list(EXAMPLES_DIR.rglob("*.ipynb"))
    return sorted(p for p in paths if ".ipynb_checkpoints" not in str(p))


def run_notebook(nb_path):
    """Execute a single notebook. Returns (success, elapsed, error_msg)."""
    nb = nbformat.read(str(nb_path), as_version=4)
    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )
    t0 = time.time()
    try:
        client.execute()
        return True, time.time() - t0, ""
    except CellExecutionError as e:
        return False, time.time() - t0, str(e)
    except Exception as e:
        return False, time.time() - t0, traceback.format_exc()


def main():
    patterns = sys.argv[1:] or None
    notebooks = find_notebooks(patterns)
    if not notebooks:
        print("No notebooks found.")
        return 1

    print(f"Found {len(notebooks)} notebook(s)\n")
    print(f"{'Notebook':<60} {'Status':>8}  {'Time':>8}")
    print("-" * 80)

    results = []
    for nb_path in notebooks:
        rel = nb_path.relative_to(EXAMPLES_DIR)
        ok, elapsed, err = run_notebook(nb_path)
        status = "OK" if ok else "FAIL"
        print(f"{str(rel):<60} {status:>8}  {elapsed:>7.1f}s")
        results.append((rel, ok, elapsed, err))

    # Summary
    passed = sum(1 for _, ok, _, _ in results if ok)
    failed = sum(1 for _, ok, _, _ in results if not ok)
    total_time = sum(t for _, _, t, _ in results)

    print("-" * 80)
    print(f"{'TOTAL':<60} {passed}P/{failed}F  {total_time:>7.1f}s")

    if failed:
        print(f"\n{'='*80}")
        print("FAILURES")
        print(f"{'='*80}")
        for rel, ok, _, err in results:
            if not ok:
                print(f"\n--- {rel} ---")
                # Truncate very long tracebacks to the last 40 lines
                lines = err.strip().splitlines()
                if len(lines) > 40:
                    lines = ["  ... (truncated) ..."] + lines[-40:]
                print("\n".join(lines))

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
