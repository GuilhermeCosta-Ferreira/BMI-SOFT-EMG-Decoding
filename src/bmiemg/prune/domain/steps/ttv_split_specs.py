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
class TTVSplitSpecs(PruneSpecs):
    name: ClassVar[str] = "ttv_split"
    train_ratio: float
    val_ratio: float
    test_ratio: float

    def __post_init__(self):
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0
