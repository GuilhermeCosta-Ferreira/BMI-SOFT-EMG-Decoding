# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar

from ..actors import DataActor, XdfData
from .prune_step import PruneStep
from .remove_session_specs import RemoveSessionSpecs



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RemoveSessionsStep(PruneStep):
    name: ClassVar[str] = "remove_sessions"
    config: RemoveSessionSpecs

    def apply(self, actors: list[XdfData]) -> list[XdfData]:
        filtered_actors = [p for p in actors if p.file_name not in self.config.files_to_remove]
        return filtered_actors
