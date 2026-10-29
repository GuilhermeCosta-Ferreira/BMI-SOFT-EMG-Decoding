# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC, abstractmethod
from pathlib import Path

from ...domain import DataActor


# ================================================================
# 1. Section: Class definition
# ================================================================
class FileLoader(ABC):
    @abstractmethod
    def load_file(self, path: Path) -> DataActor:
        raise NotImplementedError

    def load_actors(self, path: Path) -> list[DataActor]:
        """Load every actor a file holds.

        Override when one file stores several recordings (e.g. one trial per
        entry); the default wraps `load_file`.
        """
        return [self.load_file(path)]
