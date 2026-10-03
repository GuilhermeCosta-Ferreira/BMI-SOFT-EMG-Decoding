# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar

from .prune_specs import PruneSpecs


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class AnalogFilterSpecs(PruneSpecs):
    name: ClassVar[str] = "filter"
    order: int
    type: str
    cutoff: float
