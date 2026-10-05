# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar

from ..actors import XdfData
from .prune_step import PruneStep
from .protocol_definer_specs import ProtocolDefinerSpecs


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ProtocolDefinerStep(PruneStep[ProtocolDefinerSpecs, XdfData, XdfData]):
    name: ClassVar[str] = "protocol_definer"
    config: ProtocolDefinerSpecs

    def apply(self, actors: list[XdfData]) -> list[XdfData]:
        cutoffs = sorted(self.config.cuttoff_dates)

        for actor in actors:
            if actor.date is None:
                continue  # leave protocol_version at its default (undefined)
            actor.protocol_version = 1 + sum(
                1 for cutoff in cutoffs if actor.date >= cutoff
            )
        return actors
