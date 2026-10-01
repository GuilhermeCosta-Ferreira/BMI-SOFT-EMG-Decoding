# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .protocol import Protocol
from ..steps import PruneStep, PruneSpecs



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class RawArchiveProtocolV1(Protocol):
    steps: list[PruneStep] = [

    ]
    specs: list[PruneSpecs] = [

    ]
