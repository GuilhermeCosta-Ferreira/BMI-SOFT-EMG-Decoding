# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC
from typing import Any
from dataclasses import dataclass
from collections.abc import Sequence

from ..steps import PruneSpecs, PruneStep



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Protocol(ABC):
    steps: Sequence[PruneStep[Any, Any]]
    specs: Sequence[PruneSpecs]
