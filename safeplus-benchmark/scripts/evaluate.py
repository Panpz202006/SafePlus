#!/usr/bin/env python3
"""Compatibility entry point; prefer python -m safeplus evaluate."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safeplus.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
