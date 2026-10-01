# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from .prune_specs import PruneSpecs


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RemoveSessionSpecs(PruneSpecs):
    name: ClassVar[str] = "remove_sessions"
    files_to_remove: list[str]
