# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar

from ..actors import Dataset
from ..actors import TrialData

from .ttv_split_specs import TTVSplitSpecs
from .prune_step import PruneStep



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class TTVSplitStep(PruneStep):
    name: ClassVar[str] = "ttv_split"
    config: TTVSplitSpecs

    def apply(self, actors: list[TrialData]) -> list[Dataset]:
        return super().apply(actors)
