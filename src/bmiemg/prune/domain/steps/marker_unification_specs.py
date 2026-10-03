# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from .prune_specs import PruneSpecs
from .marker_schema import (
    CanonicalMarkerMap,
    SourceMarkerSchema,
)


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class MarkerUnificationSpecs(PruneSpecs):
    name: ClassVar[str] = "marker_unification"
    source_schemas: dict[int, SourceMarkerSchema]
    target: CanonicalMarkerMap
