# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar, Literal
from scipy.signal import butter, sosfiltfilt

from ..actors import TrialData
from .prune_step import PruneStep

FilterType = Literal["bandpass", "bandstop", "lowpass", "highpass"]



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class AnalogueFilterStep(PruneStep):
    name: ClassVar[str] = "analogue_filter"
    btype: FilterType
    cutoff: float | tuple[float, float]
    order: int = 1

    def __post_init__(self) -> None:
        pair = isinstance(self.cutoff, tuple)
        if self.btype in ("bandpass", "bandstop") and not pair:
            raise ValueError(f"{self.btype} needs a (low, high) cutoff pair")
        if self.btype in ("lowpass", "highpass") and pair:
            raise ValueError(f"{self.btype} needs a single cutoff value")

    def apply(self, actors: list[TrialData]) -> list[TrialData]:
        return [self._filter_actor(actor) for actor in actors]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _filter_actor(self, actor: TrialData) -> TrialData:
        cutoff = self._clamp_cutoff(actor.sfreq / 2, actor.file_name)
        sos = butter(
            self.order,
            cutoff,
            btype=self.btype,
            fs=actor.sfreq,
            output="sos",
        )

        # Zero-phase filtering keeps the signal aligned with the markers.
        filtered = sosfiltfilt(sos, actor.signal_ch, axis=1)
        return actor.copy_with(signal_ch=filtered)

    def _clamp_cutoff(
        self, nyquist: float, file_name: str
    ) -> float | tuple[float, float]:
        low = nyquist * 1e-6
        high = nyquist * (1 - 1e-6)

        def clamp(freq: float) -> float:
            return min(max(freq, low), high)

        cutoff = self.cutoff
        clamped = (
            (clamp(cutoff[0]), clamp(cutoff[1]))
            if isinstance(cutoff, tuple)
            else clamp(cutoff)
        )
        return clamped
