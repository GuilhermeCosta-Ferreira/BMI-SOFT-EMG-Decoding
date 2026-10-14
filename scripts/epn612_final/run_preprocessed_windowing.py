"""Extract annotated 200 ms windows after 50 Hz notch and 20 Hz high-pass."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from collections import Counter
from contextlib import closing
import csv
import json
import os

BASE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE / ".cache/matplotlib"))
import mne
import numpy as np
import pandas as pd
from bmiemg.data.epn612_final.preprocessed_features import (
    FEATURE_TYPES,
    _metadata_sfreq,
    _stack_emg,
    preprocess_emg,
)
from bmiemg.data.epn612_final.trees_of_trees import (
    EPN612_CHANNELS,
    _training_recordings,
)
from bmiemg.data.epoch.SignalPartitioner import split_epochs_into_windows
from bmiemg.preprocessing import TIME_FEATURE_FUNCTIONS, FREQ_FEATURE_FUNCTIONS
from bmiemg.preprocessing.features import get_emg_features
from scripts.epn612_final import run_trees_of_trees as analysis
from scripts.epn612_final import train_custom_trees_of_trees as custom
from scripts.epn612_final import make_phase2a_feature_figures as feature_plots

WINDOW_S, STEP_S = 0.2, 0.1
WINDOW_SAMPLES, STEP_SAMPLES = 40, 20
FIELDS = ("trainingSamples", "testingSamples")
NAMES = analysis.get_feature_names(EPN612_CHANNELS)


def _interval(sample, n_samples):
    """Validate supplied annotations, without signal-based segmentation or relabeling."""
    if sample["gestureName"] == "noGesture":
        if "groundTruth" in sample and np.any(sample["groundTruth"]):
            raise ValueError("noGesture contains active groundTruth labels")
        return 0, n_samples
    truth = np.asarray(sample["groundTruth"])
    bounds = sample["groundTruthIndex"]
    if truth.shape != (n_samples,) or not np.isin(truth, [0, 1]).all():
        raise ValueError("groundTruth must be a finite binary mask aligned with EMG")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(type(x) is not int for x in bounds)
    ):
        raise ValueError("groundTruthIndex must contain two integer sample indices")
    start, stop = bounds[0] - 1, bounds[1]
    if not 0 <= start < stop <= n_samples:
        raise ValueError(f"Invalid one-based inclusive groundTruthIndex: {bounds}")
    active = np.flatnonzero(truth)
    if (
        len(active) != stop - start
        or active[0] != start
        or active[-1] != stop - 1
        or not np.all(truth[start:stop] == 1)
    ):
        raise ValueError(
            "groundTruthIndex does not match the contiguous groundTruth mask"
        )
    return start, stop


def _planned_starts(n_samples, interval):
    """Plan feature storage and annotation selection; no signal segmentation here."""
    starts = np.arange(0, max(0, n_samples - WINDOW_SAMPLES + 1), STEP_SAMPLES)
    keep = (starts >= interval[0]) & (starts + WINDOW_SAMPLES <= interval[1])
    return starts, keep


def _recording_features(sample, sfreq, code):
    raw = _stack_emg(sample)
    interval = _interval(sample, raw.shape[1])
    signal = preprocess_emg(raw, sfreq)
    starts, keep = _planned_starts(raw.shape[1], interval)
    if not len(starts):
        raise ValueError("Recording is shorter than a 40-sample window")
    with mne.use_log_level("ERROR"):
        epochs = mne.EpochsArray(
            signal[None],
            mne.create_info(EPN612_CHANNELS, sfreq, ch_types="emg"),
            events=np.array([[0, 0, code + 1]]),
            event_id={sample["gestureName"]: code + 1},
            baseline=None,
            verbose=False,
        )
        windows = split_epochs_into_windows(epochs, window_s=WINDOW_S, step_s=STEP_S)
    if (
        windows.get_data(copy=False).shape != (len(starts), 8, 40)
        or not np.array_equal(windows.metadata.window_idx.to_numpy() * 20, starts)
        or not (windows.metadata.original_epoch == 0).all()
        or not (windows.events[:, 2] == code + 1).all()
    ):
        raise ValueError("Existing window utility changed shape, order, or labels")
    if keep.any():
        values = get_emg_features(
            windows.get_data(copy=False)[keep],
            sfreq,
            TIME_FEATURE_FUNCTIONS,
            FREQ_FEATURE_FUNCTIONS,
        )
    else:
        values = np.empty((0, 192), dtype=float)
    if values.shape != (int(keep.sum()), 192) or not np.isfinite(values).all():
        raise ValueError(
            "Canonical 40-sample feature extraction is non-finite or misaligned"
        )
    selected_starts = starts[keep]
    summary = {
        "sfreq": sfreq,
        "raw_shape": list(raw.shape),
        "filtered_shape": list(signal.shape),
        "candidate_windows": len(starts),
        "retained_windows": len(values),
        "discarded_windows": int((~keep).sum()),
        "interval_start": interval[0],
        "interval_stop_exclusive": interval[1],
        "first_window_start": int(selected_starts[0]) if len(values) else None,
        "unused_tail_samples": int(raw.shape[1] - starts[-1] - 40),
    }
    # Only independent feature values and scalar/shape metadata escape this call.
    del windows, epochs, signal, raw
    return values, summary


def _feature_ranges(low, high, output):
    ranges = pd.DataFrame({"feature": NAMES, "minimum": low, "maximum": high})
    ranges["constant"] = ranges.minimum == ranges.maximum
    ranges.to_csv(output, index=False)
    return ranges.loc[ranges.constant, "feature"].tolist()


def validate(output, labels):
    """Check one recording per label/field in two users before the full run."""
    blocks, summaries = [], []
    with closing(
        _training_recordings(analysis.TRAINING_DIR, analysis.ARCHIVE_PATH)
    ) as users:
        for user_index, (path, user) in enumerate(users):
            sfreq = _metadata_sfreq(user)
            for field in FIELDS:
                seen = set()
                for sample_id, sample in user[field].items():
                    label = sample["gestureName"]
                    if label in seen:
                        continue
                    seen.add(label)
                    values, summary = _recording_features(
                        sample, sfreq, labels.index(label)
                    )
                    blocks.append(values)
                    summaries.append(
                        {
                            "source": str(path),
                            "field": field,
                            "sample_id": sample_id,
                            "label": label,
                            **summary,
                        }
                    )
            if user_index == 1:
                break
    values = np.vstack(blocks)
    if values.shape[1] != 192 or len(FEATURE_TYPES) != 24 or not len(values):
        raise ValueError("Validation did not produce the full canonical feature matrix")
    constants = _feature_ranges(
        values.min(axis=0), values.max(axis=0), output / "feature_ranges.csv"
    )
    frequency = {}
    for index, fn in enumerate(
        FREQ_FEATURE_FUNCTIONS, start=len(TIME_FEATURE_FUNCTIONS)
    ):
        block = values[:, 8 * index : 8 * (index + 1)]
        if len(np.unique(block)) <= 1:
            raise ValueError(
                f"Frequency feature {fn.__name__} is degenerate across validation recordings"
            )
        if (block < 0).any() or (fn.__name__ != "total_power" and (block > 100).any()):
            raise ValueError(
                f"Frequency feature {fn.__name__} lies outside its valid range"
            )
        frequency[fn.__name__] = {
            "minimum": float(block.min()),
            "maximum": float(block.max()),
            "distinct_values": len(np.unique(block)),
        }
    report = {
        "passed": True,
        "fs_hz": 200,
        "window_ms": 200,
        "window_samples": 40,
        "step_ms": 100,
        "step_samples": 20,
        "overlap_percent": 50,
        "recordings": summaries,
        "feature_matrix_shape": list(values.shape),
        "all_24_canonical_types": True,
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "constant_columns": constants,
        "constant_feature_note": "Existing wamp/myop thresholds assume volts; supplied EMG units remain unchanged. Saturated finite values are reported and retained.",
        "frequency_features": frequency,
        "fft_bin_spacing_hz": 5,
        "labeling": "Keep complete windows within groundTruthIndex validated against groundTruth; retain all noGesture windows; never use myoDetection as truth",
        "grouping": "source user JSON + sample field + sample ID; no random window split",
        "discarded_windows": sum(s["discarded_windows"] for s in summaries),
        "waveform_files_written": 0,
    }
    (output / "validation_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "validation_summary.txt").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Validation:",
        json.dumps({k: v for k, v in report.items() if k != "recordings"}),
        flush=True,
    )
    return report


def extract(output, labels):
    """Make one file-backed canonical feature matrix, with train/test row slices."""
    counts = dict.fromkeys(FIELDS, 0)
    records, original_counts, sources = {}, dict.fromkeys(FIELDS, 0), []
    print("Auditing all annotations and planning feature storage", flush=True)
    with closing(
        _training_recordings(analysis.TRAINING_DIR, analysis.ARCHIVE_PATH)
    ) as users:
        for path, user in users:
            sources.append(str(path))
            _metadata_sfreq(user)
            for field in FIELDS:
                for sid, sample in user[field].items():
                    try:
                        lengths = [len(sample["emg"][ch]) for ch in EPN612_CHANNELS]
                        if len(set(lengths)) != 1:
                            raise ValueError("Channel sample lengths differ")
                        interval = _interval(sample, lengths[0])
                        _, keep = _planned_starts(lengths[0], interval)
                        records[(str(path), field, sid)] = int(keep.sum())
                        counts[field] += int(keep.sum())
                        original_counts[field] += 1
                    except (KeyError, ValueError, TypeError) as error:
                        raise ValueError(f"{path}, {field}, {sid}: {error}") from error
    if original_counts != {field: 45900 for field in FIELDS} or len(sources) != 306:
        raise ValueError(
            f"Original recording grouping differs from previous runs: {original_counts}"
        )
    offsets = {FIELDS[0]: 0, FIELDS[1]: counts[FIELDS[0]]}
    cursors = offsets.copy()
    features = np.lib.format.open_memmap(
        output / "features.npy",
        mode="w+",
        dtype=np.float64,
        shape=(sum(counts.values()), 192),
    )
    codes = np.lib.format.open_memmap(
        output / "label_codes.npy",
        mode="w+",
        dtype=np.int8,
        shape=(sum(counts.values()),),
    )
    low, high = np.full(192, np.inf), np.full(192, -np.inf)
    discarded, zero_window_records, class_counts = (
        0,
        0,
        {field: Counter() for field in FIELDS},
    )
    with (output / "recording_window_manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source",
                "user",
                "sample_field",
                "sample_id",
                "gesture",
                "first_feature_row",
                "windows",
                "first_window_start",
                "step_samples",
                "interval_start",
                "interval_stop_exclusive",
                "candidate_windows",
                "discarded_windows",
                "unused_tail_samples",
            ],
        )
        writer.writeheader()
        with closing(
            _training_recordings(analysis.TRAINING_DIR, analysis.ARCHIVE_PATH)
        ) as users:
            for ui, (path, user) in enumerate(users, start=1):
                sfreq = _metadata_sfreq(user)
                for field in FIELDS:
                    for sid, sample in user[field].items():
                        try:
                            label = sample["gestureName"]
                            code = labels.index(label)
                            values, summary = _recording_features(sample, sfreq, code)
                            n = len(values)
                            if n != records[(str(path), field, sid)]:
                                raise ValueError(
                                    "Window utility and annotation storage plan disagree"
                                )
                            start = cursors[field]
                            features[start : start + n] = values
                            codes[start : start + n] = code
                            cursors[field] += n
                            class_counts[field][label] += n
                            discarded += summary["discarded_windows"]
                            zero_window_records += int(n == 0)
                            if n:
                                low = np.minimum(low, values.min(axis=0))
                                high = np.maximum(high, values.max(axis=0))
                            writer.writerow(
                                {
                                    "source": str(path),
                                    "user": Path(path).parent.name,
                                    "sample_field": field,
                                    "sample_id": sid,
                                    "gesture": label,
                                    "first_feature_row": start,
                                    "windows": n,
                                    "step_samples": STEP_SAMPLES,
                                    **{
                                        k: summary[k]
                                        for k in (
                                            "first_window_start",
                                            "interval_start",
                                            "interval_stop_exclusive",
                                            "candidate_windows",
                                            "discarded_windows",
                                            "unused_tail_samples",
                                        )
                                    },
                                }
                            )
                        except (KeyError, ValueError, TypeError) as error:
                            raise ValueError(
                                f"{path}, {field}, {sid}: {error}"
                            ) from error
                if ui == 1 or ui % 10 == 0:
                    print(
                        f"Features: {ui}/306 users; train={cursors[FIELDS[0]]:,}, "
                        f"held-out={cursors[FIELDS[1]] - offsets[FIELDS[1]]:,}",
                        flush=True,
                    )
                    features.flush()
                    codes.flush()
    for field in FIELDS:
        if cursors[field] != offsets[field] + counts[field] or set(
            class_counts[field]
        ) != set(labels):
            raise ValueError(f"Incomplete window feature matrix in {field}")
    features.flush()
    codes.flush()
    constants = _feature_ranges(low, high, output / "full_feature_ranges.csv")
    report = {
        "windows": counts,
        "original_recordings": original_counts,
        "users": 306,
        "class_counts": {field: dict(class_counts[field]) for field in FIELDS},
        "discarded_windows": discarded,
        "recordings_with_no_retained_windows": zero_window_records,
        "constant_columns": constants,
        "nan_count": 0,
        "inf_count": 0,
        "waveform_files_written": 0,
        "feature_values_dtype": "float64",
        "group_identity": "source + sample_field + sample_id",
        "manifest_row_mapping": "first_feature_row + window ordinal; time start = first_window_start + ordinal * 20; all rows stay in original sample_field",
    }
    (output / "extraction_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    matrices, targets = {}, {}
    for field in FIELDS:
        start, stop = offsets[field], offsets[field] + counts[field]
        matrices[field] = pd.DataFrame(features[start:stop], columns=NAMES, copy=False)
        targets[field] = np.asarray(labels)[codes[start:stop]]
    print("Window extraction complete:", report["windows"], flush=True)
    return features, matrices, targets, report
