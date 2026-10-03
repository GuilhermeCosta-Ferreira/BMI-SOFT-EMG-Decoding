# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from .prune_specs import PruneSpecs


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SignalUnwarpSpecs(PruneSpecs):
    name: ClassVar[str] = "signal_unwarp"
    background_marker: int = 0  # value for timepoints with no marker event
