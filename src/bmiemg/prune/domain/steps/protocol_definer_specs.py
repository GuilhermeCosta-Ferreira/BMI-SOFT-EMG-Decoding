# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from datetime import date
from dataclasses import dataclass

from .prune_specs import PruneSpecs


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ProtocolDefinerSpecs(PruneSpecs):
    name: ClassVar[str] = "protocol_definer"
    cuttoff_dates: list[date]
