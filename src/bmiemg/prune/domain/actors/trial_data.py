# ================================================================
# 0. Section: IMPORTS
# ================================================================
from numpy.typing import NDArray
from dataclasses import dataclass

from .metadata_actor import MetadataActor


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass(kw_only=True)
class TrialData(MetadataActor):
    signal: NDArray  # shape (channels, time)
    markers: NDArray  # shape (1, time)
    channel_names: list[str]
    sfreq: float
