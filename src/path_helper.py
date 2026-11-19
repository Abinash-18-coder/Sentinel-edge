from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> Path:
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))
    return root_dir


PROJECT_ROOT = ensure_project_root()