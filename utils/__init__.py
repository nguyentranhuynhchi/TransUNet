"""Utils package for TransUNet with MyoPS380 benchmark adaptations."""
from __future__ import annotations

import importlib.util
from pathlib import Path

# Load original DiceLoss from root utils.py to ensure 100% preservation
_root_utils_path = Path(__file__).resolve().parent.parent / "utils.py"
if _root_utils_path.is_file():
    _spec = importlib.util.spec_from_file_location("_root_utils", str(_root_utils_path))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    DiceLoss = getattr(_mod, "DiceLoss", None)
else:
    DiceLoss = None

from . import optimization
from . import metrics
from . import efficiency

__all__ = ["DiceLoss", "optimization", "metrics", "efficiency"]
