#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
python scripts/make_synthetic.py --output /tmp/safeplus_smoke.npz --n 96 --length 12 --features 4
SAFEPLUS_SMOKE_DATA=/tmp/safeplus_smoke.npz python -m pytest -q
