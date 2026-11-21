# -*- coding: utf-8 -*-
"""``python -m vramcalc`` entry point."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())