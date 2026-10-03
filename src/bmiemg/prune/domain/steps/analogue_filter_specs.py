# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar, Literal
from dataclasses import dataclass

from .prune_specs import PruneSpecs

# ================================================================
# 1. Section: Types
# ================================================================
FilterType = Literal["bandpass", "bandstop", "lowpass", "highpass"]


# ================================================================
# 2. Section: Functions
# ================================================================
@dataclass
class AnalogueFilterSpecs(PruneSpecs):
    name: ClassVar[str] = "analogue_filter"

    btype: FilterType
    # Cutoff in Hz: a single value for low/high pass, a (low, high) pair
    # for band pass/stop.
    cutoff: float | tuple[float, float]
    order: int = 1

    def __post_init__(self) -> None:
        pair = isinstance(self.cutoff, tuple)
        if self.btype in ("bandpass", "bandstop") and not pair:
            raise ValueError(f"{self.btype} needs a (low, high) cutoff pair")
        if self.btype in ("lowpass", "highpass") and pair:
            raise ValueError(f"{self.btype} needs a single cutoff value")
