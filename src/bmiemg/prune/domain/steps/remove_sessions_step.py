# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar

from ..actors import XdfData
from .prune_step import PruneStep
from .remove_session_specs import RemoveSessionSpecs


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RemoveSessionStep(PruneStep[RemoveSessionSpecs, XdfData]):
    name: ClassVar[str] = "remove_session"

    def apply(self, actors: list[XdfData]) -> list[XdfData]:
        filtered_actors = [
            p for p in actors if p.file_name not in self.config.files_to_remove
        ]
        return filtered_actors
