# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar, Literal, get_args
from dataclasses import dataclass

from .prune_specs import PruneSpecs

# ================================================================
# 1. Section: Types
# ================================================================
Signal = Literal["emg", "eeg"]


# ================================================================
# 2. Section: Functions
# ================================================================
@dataclass
class SignalSplitSpecs(PruneSpecs):
    name: ClassVar[str] = "signal_split"
    signal: Signal

    def __post_init__(self) -> None:
        if self.signal not in get_args(Signal):
            raise ValueError(
                f"signal must be one of {get_args(Signal)}, got {self.signal!r}"
            )
