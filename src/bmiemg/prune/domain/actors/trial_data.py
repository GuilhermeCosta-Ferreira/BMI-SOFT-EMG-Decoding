# ================================================================
# 0. Section: IMPORTS
# ================================================================
from numpy.typing import NDArray
from dataclasses import dataclass

from .xdf_actor import XdfActor


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass(kw_only=True)
class TrialData(XdfActor):
    signal: NDArray  # shape (channels, time)
    channel_names: list[str]
    markers: NDArray  # shape (1, time)
