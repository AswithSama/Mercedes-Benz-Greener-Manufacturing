"""Simple project logger that doesn't create nested timestamp directories."""

import logging
from pathlib import Path

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
