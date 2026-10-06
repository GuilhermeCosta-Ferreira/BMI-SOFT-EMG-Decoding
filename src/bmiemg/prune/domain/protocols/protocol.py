# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC
from typing import Any
from dataclasses import dataclass, field
from collections.abc import Sequence

from ..steps import PruneStep


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Protocol(ABC):
    steps: Sequence[PruneStep[Any, Any]] = field(default_factory=list)
