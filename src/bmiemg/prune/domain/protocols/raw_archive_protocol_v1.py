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
    MarkerUnificationSpecs,
    MarkerUnificationStep,
    AnalogueFilterSpecs,
    AnalogueFilterStep,
)
from ..steps.marker_schema import CanonicalMarkerMap, SourceMarkerSchema
from .protocol import Protocol
from ..protocol_registry import ProtocolRegistry


# ================================================================
# 1. Section: Marker maps
# ================================================================
# Target structure (XXDD) this protocol unifies its markers into.
_CANONICAL_MARKER_MAP = CanonicalMarkerMap(
    phases={
        "undefined": 0,
        "cue": 1,
        "prep": 2,
        "move": 3,
        "return": 4,
        "iti": 5,
        "resting state, eyes open": 6,
        "resting state, eyes closed": 7,
        "start LabRecorder": 8,
        "experiment finished": 9,
        "test marker": 10,
    },
    movements={
        "undefined": 0,
        "open hand": 1,
        "close hand": 2,
        "rotate wrist right": 3,
        "rotate wrist left": 4,
        "pinch": 5,
    },
)

_SPECIAL_TRIGGERS = {
    9701: "resting state, eyes open",
    9702: "resting state, eyes closed",
    8888: "start LabRecorder",
    8899: "experiment finished",
    9999: "test marker",
}

_MARKER_SOURCE_SCHEMAS: dict[int, SourceMarkerSchema] = {
    1: SourceMarkerSchema(
        phase_pos=0,
        movement_slice=(3, 5),
        phase_map={1: "cue", 2: "prep", 3: "move", 4: "return", 5: "iti"},
        movement_groups={
            "open hand": [1, 2, 7],
            "close hand": [3, 4, 5, 6, 8, 15, 16],
            "rotate wrist right": [11, 12],
            "rotate wrist left": [13, 14],
            "pinch": [17],
        },
        special_triggers=_SPECIAL_TRIGGERS,
    ),
    2: SourceMarkerSchema(
        phase_pos=0,
        movement_slice=(3, 5),
        phase_map={1: "cue", 3: "prep", 5: "move", 7: "return", 9: "iti"},
        movement_groups={
            "open hand": [1],
            "close hand": [2, 3, 10, 11],
            "rotate wrist right": [4, 5],
            "rotate wrist left": [8, 9],
            "pinch": [12],
        },
        special_triggers=_SPECIAL_TRIGGERS,
    ),
}


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
        analogue_filter = AnalogueFilterSpecs(
            btype="bandpass", cutoff=(20.0, 500.0), order=1
        )
        marker_unification = MarkerUnificationSpecs(
            source_schemas=_MARKER_SOURCE_SCHEMAS,
            target=_CANONICAL_MARKER_MAP,
        )

        self.specs = [
            remove_sessions,
            signal_split,
            protocol_definer,
            signal_unwarp,
            analogue_filter,
            marker_unification,
        ]
        self.steps = [
            RemoveSessionStep(remove_sessions),
            SignalSplitStep(signal_split),
            ProtocolDefinerStep(protocol_definer),
            SignalUnwarpStep(signal_unwarp),
            AnalogueFilterStep(analogue_filter),
            MarkerUnificationStep(marker_unification),
        ]
