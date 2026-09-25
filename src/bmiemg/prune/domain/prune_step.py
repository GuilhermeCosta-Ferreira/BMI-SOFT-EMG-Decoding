# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC, abstractmethod
from typing import ClassVar
from dataclasses import dataclass

from .prune_specs import PruneSpecs
from .data_actor import DataActor



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PruneStep(ABC):
    name: ClassVar[str]
    config: PruneSpecs

    @abstractmethod
    def apply(self, actors: list[DataActor]) -> list[DataActor]:
        pass
