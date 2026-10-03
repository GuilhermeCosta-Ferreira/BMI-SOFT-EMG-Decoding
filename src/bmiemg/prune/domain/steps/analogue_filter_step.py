# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass
from scipy.signal import butter, sosfiltfilt

from ..actors import TrialData
from .prune_step import PruneStep
from .analogue_filter_specs import AnalogueFilterSpecs



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class AnalogueFilterStep(PruneStep[AnalogueFilterSpecs, TrialData, TrialData]):
    name: ClassVar[str] = "analogue_filter"
    config: AnalogueFilterSpecs

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [self._filter_actor(actor) for actor in actors]


    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _filter_actor(self, actor: TrialData) -> TrialData:
        cutoff = self._clamp_cutoff(actor.sfreq / 2, actor.file_name)
        sos = butter(
            self.config.order,
            cutoff,
            btype=self.config.btype,
            fs=actor.sfreq,
            output="sos",
        )
        # Zero-phase filtering keeps the signal aligned with the markers.
        filtered = sosfiltfilt(sos, actor.signal, axis=1)
        return actor.copy_with(signal=filtered)

    def _clamp_cutoff(
        self, nyquist: float, file_name: str
    ) -> float | tuple[float, float]:
        low = nyquist * 1e-6
        high = nyquist * (1 - 1e-6)

        def clamp(freq: float) -> float:
            return min(max(freq, low), high)

        cutoff = self.config.cutoff
        clamped = (
            (clamp(cutoff[0]), clamp(cutoff[1]))
            if isinstance(cutoff, tuple)
            else clamp(cutoff)
        )
        return clamped
