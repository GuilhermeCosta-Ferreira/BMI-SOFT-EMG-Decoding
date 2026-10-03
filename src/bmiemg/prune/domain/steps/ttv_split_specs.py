# ================================================================
# 0. Section: IMPORTS
# ================================================================
import math
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
    user_balance: bool = False
    movement_balance: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not math.isclose(total, 1.0):
            raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")
