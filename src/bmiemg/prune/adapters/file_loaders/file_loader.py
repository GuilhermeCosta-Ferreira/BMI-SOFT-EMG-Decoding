# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from abc import ABC, abstractmethod

from ...domain import DataActor


# ================================================================
# 1. Section: Class definition
# ================================================================
class FileLoader(ABC):
    @abstractmethod
    def load_file(self, path: Path) -> DataActor:
        raise NotImplementedError

    def load_actors(self, path: Path) -> list[DataActor]:
        return [self.load_file(path)]
