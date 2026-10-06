# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..actors import DataActor


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PruneStep[InT: DataActor, OutT: DataActor](ABC):
    name: ClassVar[str]

    @abstractmethod
    def apply(self, actors: list[InT]) -> list[OutT]:
        pass
