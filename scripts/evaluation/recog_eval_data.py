"""Load one XDF recording and make labeled EMG windows for offline evaluation."""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bmiemg.data.convert import session_load
from bmiemg.data.epoch import V2_TRIGGER_MAP


CHANNELS = ("AUX7", "AUX12", "AUX8", "AUX11", "AUX10")


@dataclass(frozen=True)
class Trial:
    number: int
    code: int
    label: str
    prep_start: float
    start: float
    end: float


@dataclass(frozen=True)
class BatchInfo:
    time: float
    trial: int
    phase: str
    truth: str | None


# ================================================================
# 1. Map an XDF movement code to its gesture label
# ================================================================
# Search the existing trigger map for the code.
# Raise an error if the code has no gesture label.
def movement_label(code: int) -> str:
    for label, codes in V2_TRIGGER_MAP.target_code.items():
        if code in codes:
            return label
    raise ValueError(f"Movement code {code} is not mapped by V2_TRIGGER_MAP")


# ================================================================
# 2. Rebuild trials from preparation, movement, and return markers
# ================================================================
# Read the XDF marker stream in chronological order.
# Match markers 3, 5, and 7 for each movement.
# Store the gesture and its preparation and movement times.
def extract_trials(marker_stream) -> list[Trial]:
    trials = []
    prepared = None
    moving = None

    for timestamp, value in zip(marker_stream.time_stamps, marker_stream.time_series):
        marker = str(value[0]).strip()
        if len(marker) != 5 or not marker.isdigit():
            continue
        phase, identity = marker[0], marker[1:]

        if phase == "3" and moving is None:
            prepared = (float(timestamp), identity)
        elif phase == "5" and prepared is not None and prepared[1] == identity and moving is None:
            moving = (float(timestamp), prepared[0], identity)
            prepared = None
        elif phase == "7" and moving is not None:
            start, prep_start, current_identity = moving
            if identity != current_identity:
                raise ValueError(f"Return marker {marker} does not match movement")
            code = int(identity[-2:])
            trials.append(
                Trial(len(trials) + 1, code, movement_label(code), prep_start,
                      start, float(timestamp))
            )
            moving = None

    if moving is not None or not trials:
        raise ValueError("Incomplete or missing movement trials in XDF markers")
    return trials


# ================================================================
# 3. Split each code's trials into training and testing
# ================================================================
# Group trials by movement code.
# Use the first train_per_code trials for training.
# Put later trials in the held-out test set.
def split_trials(trials: list[Trial], train_per_code: int) -> tuple[set[int], set[int]]:
    by_code = defaultdict(list)
    for trial in trials:
        by_code[trial.code].append(trial.number)

    train, test = set(), set()
    for code, numbers in by_code.items():
        if len(numbers) <= train_per_code:
            raise ValueError(
                f"Code {code:02d} has {len(numbers)} trials; need more than "
                f"{train_per_code} for a held-out test"
            )
        train.update(numbers[:train_per_code])
        test.update(numbers[train_per_code:])
    return train, test


# ================================================================
# 4. Find each trial's end and the following rest period
# ================================================================
# Find marker 9 after each movement's return marker.
# Label rest from marker 9 until the next preparation marker.
# Leave the final rest period unlabeled because its end is unknown.
def trial_bounds(marker_stream, trials: list[Trial]) -> dict:
    events = [
        (float(time), str(value[0]).strip())
        for time, value in zip(marker_stream.time_stamps, marker_stream.time_series)
    ]
    bounds = {}
    for index, trial in enumerate(trials):
        next_prep = trials[index + 1].prep_start if index + 1 < len(trials) else None
        iti = next(
            (
                time
                for time, marker in events
                if time > trial.end
                and (next_prep is None or time < next_prep)
                and len(marker) == 5
                and marker[0] == "9"
                and marker.isdigit()
                and int(marker[-2:]) == trial.code
            ),
            None,
        )
        if iti is None:
            raise ValueError(f"Missing ITI marker after trial {trial.number}")
        # The last ITI has no known end, so it is not labeled as rest.
        bounds[trial.number] = (next_prep or iti, (iti, next_prep) if next_prep else None)
    return bounds


# ================================================================
# 5. Load the XDF and convert selected EMG channels to volts
# ================================================================
# Load the recording once and extract its trials and rest bounds.
# Keep the five model channels in their expected order.
# Return EMG, timestamps, sampling rate, trials, and bounds.
def load_recording(path: Path):
    session = session_load(path)
    trials = extract_trials(session.marker_stream)
    bounds = trial_bounds(session.marker_stream, trials)
    stream = session.signal_stream
    names = list(stream.channel_names)
    missing = set(CHANNELS) - set(names)
    if missing:
        raise ValueError(f"Missing EMG channels: {sorted(missing)}")
    indices = [names.index(name) for name in CHANNELS]
    # Match SignalStream.to_raw(): recorded microvolts -> volts.
    emg_volts = stream.time_series[:, indices] / 2 * 1e-6
    return emg_volts, stream.time_stamps, float(stream.sfreq), trials, bounds


# ================================================================
# 6. Make fixed-size batches with or without overlap
# ================================================================
# Convert window and step durations from milliseconds to samples.
# A full-window step gives no overlap; a smaller step overlaps.
# Label only batches fully inside movement or known rest.
# Keep other batches for raw temporal recognition.
def make_batches(
    emg_volts: np.ndarray,
    timestamps: np.ndarray,
    sfreq: float,
    trials: list[Trial],
    bounds: dict,
    selected_trials: set[int],
    window_ms: int,
    step_ms: int,
) -> tuple[np.ndarray, list[BatchInfo]]:
    window_samples = round(sfreq * window_ms / 1000)
    step_samples = round(sfreq * step_ms / 1000)
    if window_samples < 3 or not 0 < step_samples <= window_samples:
        raise ValueError("Use a positive step no larger than the window")
    if not np.isclose(window_samples / sfreq * 1000, window_ms, atol=0.5):
        raise ValueError("Window duration cannot be represented at this sample rate")
    if not np.isclose(step_samples / sfreq * 1000, step_ms, atol=0.5):
        raise ValueError("Step duration cannot be represented at this sample rate")

    batches, infos = [], []
    for trial in trials:
        if trial.number not in selected_trials:
            continue
        trial_end, rest = bounds[trial.number]
        first = int(np.searchsorted(timestamps, trial.prep_start, side="left"))
        last = int(np.searchsorted(timestamps, trial_end, side="left"))
        for offset in range(first, last - window_samples + 1, step_samples):
            batch_start = float(timestamps[offset])
            batch_end = float(timestamps[offset + window_samples - 1] + 1 / sfreq)
            if batch_start >= trial.start and batch_end <= trial.end:
                phase, truth = "movement", trial.label
            elif rest and batch_start >= rest[0] and batch_end <= rest[1]:
                phase, truth = "rest", "noGesture"
            else:
                phase, truth = "unlabeled", None
            batches.append(emg_volts[offset : offset + window_samples].T)
            infos.append(BatchInfo(batch_start, trial.number, phase, truth))

    if not batches:
        raise ValueError("No complete batches were found")
    return np.stack(batches), infos
