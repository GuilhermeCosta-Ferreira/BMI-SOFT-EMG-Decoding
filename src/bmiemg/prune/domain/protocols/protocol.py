# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC
from dataclasses import dataclass

from ..steps import PruneStep, PruneSpecs



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Protocol(ABC):
    steps: list[PruneStep]
    specs: list[PruneSpecs]
