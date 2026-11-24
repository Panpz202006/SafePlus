"""Allocate isolated directories for generated experiment results."""

from datetime import datetime, timezone
from pathlib import Path
from tempfile import mkdtemp


def new_run_directory(*parts):
    """Create an ignored run directory without overwriting prior results."""
    root = Path(__file__).resolve().parent.joinpath("runs", *parts)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(mkdtemp(prefix=f"{timestamp}-", dir=root))
