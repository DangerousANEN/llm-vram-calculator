#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy CLI entry point.

This file used to hold the whole calculator, including a parameter-count
heuristic and a benchmark that reported stream *chunks* as tokens per second.
Both are gone; the real implementation lives in the ``vramcalc`` package.

Running ``python vram_calc.py ...`` still works (arguments are passed through)
but prints a one-line migration notice on stderr. ``--json`` output on stdout
stays machine readable.
"""
from __future__ import annotations

import sys

MIGRATION = (
    "[deprecated] vram_calc.py delegates to the vramcalc package. "
    "Prefer `python -m vram_calc ...` (or the installed `vram-calc` command). "
    "The old --params heuristic requires --legacy-heuristic and is no longer exact."
)


def main(argv=None) -> int:
    from vramcalc.cli import main as cli_main

    print(MIGRATION, file=sys.stderr)
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
