# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from typing import ClassVar
from abc import ABC, abstractmethod

from ...domain.actors import DatasetPartition



# ================================================================
# 1. Section: Class definition
# ================================================================
class Writer(ABC):
    suffix: ClassVar[str]

    @abstractmethod
    def write(self, partition: DatasetPartition, dest_stem: Path) -> None:
        raise NotImplementedError
