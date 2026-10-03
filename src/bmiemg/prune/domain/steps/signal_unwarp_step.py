# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from .prune_step import PruneStep
from ..actors import XdfData, TrialData
from .signal_unwarp_specs import SignalUnwarpSpecs



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SignalWarpStep(PruneStep):
    name: ClassVar[str] = "signal_unwarp"
    config: SignalUnwarpSpecs

    def apply(self, actors: list[XdfData]) -> list[TrialData]:
        return super().apply(actors)
