# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC, abstractmethod
from typing import ClassVar
from dataclasses import dataclass

from .data_actor import DataActor
from .prune_step import PruneStep



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Protocol(ABC):
    name: ClassVar

    @abstractmethod
    def apply(self, actors: list[DataActor]) -> list[DataActor]:
        raise NotImplementedError
