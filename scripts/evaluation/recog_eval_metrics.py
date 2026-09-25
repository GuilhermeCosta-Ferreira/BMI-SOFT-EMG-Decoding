"""Score raw batch predictions without smoothing or voting."""

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from recog_eval_data import BatchInfo, Trial


# ================================================================
# 1. Write predictions from both modes to one CSV
# ================================================================
# Save each batch's mode, time, trial, phase, truth, and prediction.
# Leave the ground-truth field empty for unlabeled batches.
def write_predictions(path: Path, rows: list[tuple[str, BatchInfo, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("mode", "xdf_time_s", "trial", "phase", "ground_truth", "prediction"))
        for mode, info, prediction in rows:
            writer.writerow((mode, info.time, info.trial, info.phase,
                             info.truth or "", prediction))


# ================================================================
# 2. Print a one-vs-all percentage matrix for each gesture
# ================================================================
# Treat one gesture as positive and all other labels as negative.
# Count true positives, false negatives, false positives,
# and true negatives, then print percentages for each reality row.
def print_gesture_matrices(truth: np.ndarray, predictions: np.ndarray) -> None:
    print("Gesture matrices: positive=this gesture, negative=all other labels")
    print("Each reality row adds up to 100%.")

    for gesture in sorted(set(truth) - {"noGesture"}):
        real_positive = truth == gesture
        predicted_positive = predictions == gesture

        tp = np.count_nonzero(real_positive & predicted_positive)
        fn = np.count_nonzero(real_positive & ~predicted_positive)
        fp = np.count_nonzero(~real_positive & predicted_positive)
        tn = np.count_nonzero(~real_positive & ~predicted_positive)

        print(f"\n{gesture}:")
        print(f"{'Reality / prediction':>22} {'Positive predicted':>20} {'Negative predicted':>20}")
        print(f"{'Positive real':>22} {tp / (tp + fn):>17.1%} TP {fn / (tp + fn):>17.1%} FN")
        print(f"{'Negative real':>22} {fp / (fp + tn):>17.1%} FP {tn / (fp + tn):>17.1%} TN")


# ================================================================
# 3. Count trials with one correct raw gesture and rho > 0.25
# ================================================================
# Turn successive non-rest predictions into gesture segments.
# Require exactly one segment with the trial's true gesture.
# Compare its time span with the XDF movement markers using rho.
def raw_recognition_count(
    trials: list[Trial],
    test_trials: set[int],
    infos: list[BatchInfo],
    predictions: np.ndarray,
    window_ms: int,
) -> int:
    recognized = 0
    window_seconds = window_ms / 1000

    for trial in trials:
        if trial.number not in test_trials:
            continue

        segments = []
        previous = "noGesture"
        for info, prediction in zip(infos, predictions):
            if info.trial != trial.number:
                continue
            if prediction != "noGesture":
                if prediction != previous:
                    segments.append([prediction, info.time, info.time + window_seconds])
                else:
                    segments[-1][2] = info.time + window_seconds
            previous = prediction

        if len(segments) != 1 or segments[0][0] != trial.label:
            continue

        _, predicted_start, predicted_end = segments[0]
        overlap = max(
            0.0,
            min(trial.end, predicted_end) - max(trial.start, predicted_start),
        )
        rho = 2 * overlap / (
            (trial.end - trial.start) + (predicted_end - predicted_start)
        )
        if rho > 0.25:
            recognized += 1

    return recognized


# ================================================================
# 4. Report classification and raw recognition for one mode
# ================================================================
# Score labeled movement and rest batches for classification.
# Print accuracy, class scores, confusion matrices, and rest alarms.
# Count raw temporal recognition using all batches, even unlabeled.
# Return the main rates for the final mode comparison.
def report_predictions(
    mode: str,
    trials: list[Trial],
    test_trials: set[int],
    infos: list[BatchInfo],
    predictions: np.ndarray,
    window_ms: int,
) -> dict:
    scored = np.asarray([info.truth is not None for info in infos])
    truth = np.asarray([info.truth for info in infos if info.truth is not None])
    scored_predictions = predictions[scored]
    classes = sorted(set(truth) | set(predictions))
    accuracy = accuracy_score(truth, scored_predictions)
    rest = truth == "noGesture"
    false_gesture_rate = np.mean(scored_predictions[rest] != "noGesture")
    recognized = raw_recognition_count(
        trials, test_trials, infos, predictions, window_ms
    )

    print(f"\n{mode}: {len(infos)} test batches ({len(truth)} labeled)")
    print(f"Classification accuracy: {accuracy:.1%}")
    print(classification_report(truth, scored_predictions, labels=classes, zero_division=0))
    print("Confusion matrix (rows=truth, columns=prediction):")
    print(confusion_matrix(truth, scored_predictions, labels=classes))
    print_gesture_matrices(truth, scored_predictions)
    print(f"\nFalse-gesture rate during labeled rest: {false_gesture_rate:.1%}")
    print(f"Raw temporal recognition: {recognized}/{len(test_trials)} trials")

    return {
        "labeled_batches": len(truth),
        "classification": accuracy,
        "recognition": recognized / len(test_trials),
        "false_gesture": false_gesture_rate,
    }
