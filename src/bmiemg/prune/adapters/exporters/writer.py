# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from abc import ABC, abstractmethod

from ...domain.actors import DataActor
from .export_context import ExportContext


# ================================================================
# 1. Section: Class definition
# ================================================================
class Writer[A: DataActor](ABC):
    @abstractmethod
    def write(self, actor: A, dest_dir: Path, ctx: ExportContext) -> None:
        raise NotImplementedError
