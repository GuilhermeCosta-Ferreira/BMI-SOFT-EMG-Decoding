# ================================================================
# 0. Section: IMPORTS
# ================================================================
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..steps import PruneSpecs, PruneStep, RemoveSessionSpecs, RemoveSessionStep
from .protocol import Protocol
from ..protocol_registry import ProtocolRegistry



# ================================================================
# 1. Section: Functions
# ================================================================
@ProtocolRegistry.register("raw_archive_v1")
@dataclass
class RawArchiveProtocolV1(Protocol):
    steps: Sequence[PruneStep[Any, Any]] = field(default_factory=list)
    specs: Sequence[PruneSpecs] = field(default_factory=list)

    def __post_init__(self) -> None:
        remove_sessions = RemoveSessionSpecs(files_to_remove=[""])
        self.specs = [remove_sessions]

        self.steps = [RemoveSessionStep(remove_sessions)]
