# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar

from .prune_step import PruneStep
from ..actors import TrialData, XdfData


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class FilterStep(PruneStep):
    name: ClassVar[str] = "filter"

    def apply(self, actors: list[TrialData | XdfData]) -> list[TrialData | XdfData]:
        return super().apply(actors)
