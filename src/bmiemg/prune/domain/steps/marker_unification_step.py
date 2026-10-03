# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from .prune_step import PruneStep
from ..actors import XdfData



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MarkerUnificationStep(PruneStep):
    name: ClassVar[str] = "marker_unification"

    def apply(self, actors: list[XdfData]) -> list[XdfData]:
        return super().apply(actors)
