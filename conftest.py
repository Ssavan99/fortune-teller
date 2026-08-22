"""Put the repository root on ``sys.path`` so ``src`` imports without an install step."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
