# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Any
from datetime import date
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..steps import (
    PruneSpecs,
    PruneStep,
    RemoveSessionSpecs,
    RemoveSessionStep,
    SignalSplitStep,
    SignalSplitSpecs,
    ProtocolDefinerStep,
    ProtocolDefinerSpecs,
    SignalUnwarpSpecs,
    SignalUnwarpStep,
)
from .protocol import Protocol
from ..protocol_registry import ProtocolRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@ProtocolRegistry.register("raw_archive_v1")
@dataclass
class RawArchiveProtocolV1(Protocol):
    steps: Sequence[PruneStep[Any, Any, Any]] = field(default_factory=list)
    specs: Sequence[PruneSpecs] = field(default_factory=list)

    def __post_init__(self) -> None:
        remove_sessions = RemoveSessionSpecs(
            files_to_remove=[
                "sub-05_ses-04_task-Down_run-01_raw",
                "sub-P008_ses-S001_task-Default_run-001_emg_kraken",
                "sub-P008_ses-S002_task-Default_run-001_emg_kraken",
                "sub-P008_ses-S003_task-Default_run-001_emg_kraken",
            ]
        )
        signal_split = SignalSplitSpecs(signal="emg")
        protocol_definer = ProtocolDefinerSpecs(
            cuttoff_dates=[date(2025, 11, 21)]
        )
        signal_unwarp = SignalUnwarpSpecs()

        self.specs = [remove_sessions, signal_split, protocol_definer, signal_unwarp]
        self.steps = [RemoveSessionStep(remove_sessions), SignalSplitStep(signal_split), ProtocolDefinerStep(protocol_definer), SignalUnwarpStep(signal_unwarp)]
