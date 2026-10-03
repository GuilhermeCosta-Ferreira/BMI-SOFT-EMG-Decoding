# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import Any, ClassVar
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..actors import XdfData, TrialData
from .prune_step import PruneStep
from .signal_unwarp_specs import SignalUnwarpSpecs

_MARKERS_TYPE = "markers"


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SignalUnwarpStep(PruneStep[SignalUnwarpSpecs, XdfData, TrialData]):
    name: ClassVar[str] = "signal_unwarp"
    config: SignalUnwarpSpecs

    def apply(self, actors: list[XdfData]) -> list[TrialData]:
        return [self._unwarp_actor(actor) for actor in actors]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _unwarp_actor(self, actor: XdfData) -> TrialData:
        signal_stream, marker_stream = _split_streams(actor.streams)

        signal, channel_names = _build_signal(signal_stream)
        markers = self._build_markers(marker_stream, signal_stream)

        return TrialData(
            file_header=actor.file_header,
            file_name=actor.file_name,
            file_path=actor.file_path,
            directory_structure=actor.directory_structure,
            protocol_version=actor.protocol_version,
            signal=signal,
            channel_names=channel_names,
            markers=markers,
        )

    def _build_markers(
        self, marker_stream: dict, signal_stream: dict
    ) -> NDArray:
        signal_ts = np.asarray(signal_stream["time_stamps"], dtype=float)
        markers = np.full((1, signal_ts.size), self.config.background_marker, dtype=int)

        marker_ts = np.asarray(marker_stream["time_stamps"], dtype=float)
        for value, timestamp in zip(marker_stream["time_series"], marker_ts):
            sample = _nearest_sample(signal_ts, timestamp)
            markers[0, sample] = _marker_value(value)

        return markers


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Module Helpers
# ──────────────────────────────────────────────────────
def _split_streams(streams: list[dict]) -> tuple[dict, dict]:
    signal_stream: dict | None = None
    marker_stream: dict | None = None

    for stream in streams:
        if _stream_type(stream) == _MARKERS_TYPE:
            marker_stream = stream
        else:
            signal_stream = stream

    if signal_stream is None or marker_stream is None:
        raise ValueError(
            "signal_unwarp expects exactly one signal stream and one marker stream"
        )

    return signal_stream, marker_stream


def _build_signal(signal_stream: dict) -> tuple[NDArray, list[str]]:
    # Raw time_series is (time, channels); the signal matrix is (channels, time).
    signal = np.asarray(signal_stream["time_series"]).T

    channels = signal_stream["info"]["desc"][0]["channels"][0]["channel"]
    channel_names = [str(chan["label"][0]) for chan in channels]

    return signal, channel_names


def _nearest_sample(signal_ts: NDArray, timestamp: float) -> int:
    right = int(np.searchsorted(signal_ts, timestamp))
    if right == 0:
        return 0
    if right >= signal_ts.size:
        return signal_ts.size - 1

    left = right - 1
    if timestamp - signal_ts[left] <= signal_ts[right] - timestamp:
        return left
    return right


def _marker_value(raw: Any) -> int:
    if isinstance(raw, (list, tuple, np.ndarray)):
        raw = raw[0]
    return int(float(str(raw).strip()))


def _stream_type(stream: dict) -> str | None:
    value = stream.get("info", {}).get("type")
    if not value:
        return None
    return str(value[0]).lower()
