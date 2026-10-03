# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .prune_specs import PruneSpecs
from ..actors import DataActor


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PruneStep[SpecT: PruneSpecs, InT: DataActor, OutT: DataActor](ABC):
    name: ClassVar[str]
    config: SpecT

    @abstractmethod
    def apply(self, actors: list[InT]) -> list[OutT]:
        pass
