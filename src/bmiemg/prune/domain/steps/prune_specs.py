# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC
from typing import ClassVar
from dataclasses import dataclass



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class PruneSpecs(ABC):
    name: ClassVar[str]
