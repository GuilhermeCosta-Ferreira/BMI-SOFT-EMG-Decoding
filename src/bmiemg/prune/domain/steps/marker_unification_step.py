# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

import numpy as np

from ..actors import TrialData
from .prune_step import PruneStep
from .marker_unification_specs import MarkerUnificationSpecs



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MarkerUnificationStep(PruneStep[MarkerUnificationSpecs, TrialData, TrialData]):
    name: ClassVar[str] = "marker_unification"
    config: MarkerUnificationSpecs

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [self._unify_actor(actor) for actor in actors]


    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _unify_actor(self, actor: TrialData) -> TrialData:
        schema = self.config.source_schemas.get(actor.protocol_version)
        if schema is None:
            raise ValueError(
                f"No source marker schema for protocol version "
                f"{actor.protocol_version!r} ({actor.file_name})"
            )

        markers = actor.markers
        unified = markers.copy()
        # Remap each distinct raw code once, in place across every sample.
        for raw in np.unique(markers):
            raw = int(raw)
            if raw == 0:  # background stays undefined (0000)
                continue
            phase, movement = schema.decode(raw)
            unified[markers == raw] = self.config.target.encode(phase, movement)

        return actor.copy_with(markers=unified)
