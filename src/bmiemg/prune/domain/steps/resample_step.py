# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from fractions import Fraction
from typing import ClassVar, Literal

import numpy as np
from scipy.signal import resample_poly

from ..actors import TrialData
from .prune_step import PruneStep
from .step_registry import StepRegistry

PadType = Literal["constant", "line", "mean", "median", "maximum", "minimum"]


# ================================================================
# 1. Section: Functions
# ================================================================
@StepRegistry.register("resample")
@dataclass
class ResampleStep(PruneStep[TrialData, TrialData]):
    name: ClassVar[str] = "resample"
    sfreq: float
    padtype: PadType = "constant"

    def __post_init__(self) -> None:
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}")

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [self._resample_actor(actor) for actor in actors]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _resample_actor(self, actor: TrialData) -> TrialData:
        ratio = Fraction(self.sfreq / actor.sfreq).limit_denominator(100_000)
        up, down = ratio.numerator, ratio.denominator
        if up == down:
            return actor

        signal = resample_poly(actor.signal_ch, up, down, axis=1, padtype=self.padtype)

        # Markers are labels, not signal: hold each source sample's code, so an
        # onset at source index i lands on i * up / down.
        n_source = actor.markers.shape[1]
        source_idx = (np.arange(signal.shape[1]) * down) // up
        markers = actor.markers[:, np.minimum(source_idx, n_source - 1)]

        return actor.copy_with(signal_ch=signal, markers=markers, sfreq=self.sfreq)
