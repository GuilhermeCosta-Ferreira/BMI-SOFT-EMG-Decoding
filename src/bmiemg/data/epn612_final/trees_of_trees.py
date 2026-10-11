"""EPN612 inputs for one-versus-all TreesOfTrees analyses."""

import json
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence
from zipfile import ZipFile

import numpy as np
import pandas as pd

from bmiemg.postprocessing.importance_pruning import get_feature_df
from bmiemg.preprocessing import FREQ_FEATURE_FUNCTIONS, TIME_FEATURE_FUNCTIONS
from bmiemg.preprocessing.features import get_emg_features

EPN612_CHANNELS = tuple(f"ch{i}" for i in range(1, 9))


def _training_recordings(training_dir: Path, archive_path: Path | None = None):
    """Read the local archive when supplied, avoiding offloaded extracted files."""
    if archive_path is not None and archive_path.is_file():
        prefix = f"{training_dir.parent.name}/{training_dir.name}/"
        with ZipFile(archive_path) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.startswith(prefix) and Path(name).match("*/user*/user*.json")
            )
            if not members:
                raise FileNotFoundError(
                    f"No EPN612 user JSON files found in {archive_path}"
                )
            for name in members:
                with archive.open(name) as file:
                    yield name, json.load(file)
        return

    paths = sorted(training_dir.glob("user*/user*.json"))
    if not paths:
        raise FileNotFoundError(f"No EPN612 user JSON files found in {training_dir}")
    for path in paths:
        with path.open(encoding="utf-8") as file:
            yield path, json.load(file)


# ================================================================
# 1. Section: Validate selected channels
# ================================================================
# This function checks that each requested channel is a unique EPN612 EMG channel.
# It returns the channels in the exact order supplied by the caller.
def validate_channels(channels: Sequence[str]) -> tuple[str, ...]:
    """Validate a caller-supplied channel order without changing it."""
    if isinstance(channels, str):
        raise ValueError("channels must be an ordered sequence, not a string")
    selected = tuple(channels)
    if not selected:
        raise ValueError("Select at least one EPN612 EMG channel")
    invalid = [channel for channel in selected if channel not in EPN612_CHANNELS]
    if invalid:
        raise ValueError(f"Unknown EPN612 EMG channels: {invalid}; expected ch1–ch8")
    if len(selected) != len(set(selected)):
        raise ValueError("EPN612 EMG channels must not be repeated")
    return selected


# ================================================================
# 2. Section: Name feature columns
# ================================================================
# This function labels each column using the existing feature list and naming helper.
# It keeps the extractor's feature-major order and the caller's channel order.
def get_feature_names(channels: Sequence[str]) -> list[str]:
    """Use the existing feature-major naming order used by the extractor."""
    selected = validate_channels(channels)
    feature_functions = [*TIME_FEATURE_FUNCTIONS, *FREQ_FEATURE_FUNCTIONS]
    # get_feature_df only reads ch_names; no MNE signal object is needed here.
    channel_info = SimpleNamespace(ch_names=selected)
    return get_feature_df(channel_info, feature_functions)["feature"].tolist()


# ================================================================
# 3. Section: Discover gesture labels
# ================================================================
# This function reads the EPN612 gesture mapping from a training user file.
# It returns the labels in the order of their dataset codes.
def get_training_labels(
    training_dir: Path, archive_path: Path | None = None
) -> list[str]:
    """Discover gestures from the EPN612 dataset's own label mapping."""
    with closing(_training_recordings(training_dir, archive_path)) as recordings:
        _, recording = next(recordings)
    label_codes = recording["generalInfo"]["myoPredictionLabel"]
    return sorted(label_codes, key=label_codes.get)


# ================================================================
# 4. Section: Extract training features
# ================================================================
# This function reads labeled EPN612 recordings in the selected channel order.
# It extracts all canonical features and returns aligned feature rows and labels.
def load_training_features(
    training_dir: Path,
    channels: Sequence[str],
    archive_path: Path | None = None,
    sample_field: str = "trainingSamples",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Read labeled EPN612 recordings and extract canonical features.

    ``sample_field`` selects the supplied training or held-out testing recordings
    inside each training-partition user file. Each five-second recording is one
    example; no additional preprocessing or windowing is applied.
    """
    if sample_field not in {"trainingSamples", "testingSamples"}:
        raise ValueError(f"Unsupported EPN612 sample field: {sample_field}")
    selected = validate_channels(channels)
    names = get_feature_names(selected)
    feature_blocks = []
    labels = []
    sampling_frequency = None
    for path, recording in _training_recordings(training_dir, archive_path):

        sfreq = float(recording["generalInfo"]["samplingFrequencyInHertz"])
        if sampling_frequency is None:
            sampling_frequency = sfreq
        elif sfreq != sampling_frequency:
            raise ValueError(f"Sampling frequency differs in {path}: {sfreq}")

        samples = recording[sample_field]
        block = np.empty((len(samples), len(names)), dtype=float)
        for row, (sample_id, sample) in enumerate(samples.items()):
            try:
                signal = np.stack([sample["emg"][ch] for ch in selected])
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"Invalid EMG channels in {path}, {sample_id}"
                ) from error
            values = get_emg_features(
                signal[None, :, :],
                sfreq,
                TIME_FEATURE_FUNCTIONS,
                FREQ_FEATURE_FUNCTIONS,
            )[0]
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite features in {path}, {sample_id}")
            block[row] = values
            if "gestureName" not in sample:
                raise ValueError(f"Unlabeled sample in {path}, {sample_id}")
            labels.append(sample["gestureName"])
        feature_blocks.append(block)

    features = pd.DataFrame(np.vstack(feature_blocks), columns=names)
    return features, np.asarray(labels)
