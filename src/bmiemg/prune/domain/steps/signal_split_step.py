# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from ..actors import XdfData
from .prune_step import PruneStep
from .signal_split_specs import SignalSplitSpecs



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SignalSplitStep(PruneStep[SignalSplitSpecs, XdfData]):
    name: ClassVar[str]
    config: SignalSplitSpecs

    def apply(self, actors: list[XdfData]) -> list[XdfData]:
