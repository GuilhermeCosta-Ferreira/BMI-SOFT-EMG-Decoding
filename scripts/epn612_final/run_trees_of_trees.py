"""Shared tree evaluation and importance helpers for cached final features."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import os

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib")
)

import json
from types import SimpleNamespace
from typing import Sequence
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from bmiemg.data.epn612_final.trees_of_trees import (
    EPN612_CHANNELS,
    get_feature_names,
    get_training_labels,
    validate_channels,
)
from bmiemg.download.adapters.emg_epn_612 import EPN612Specs
from bmiemg.models.model_factories import DecisionTreeFactory
from bmiemg.postprocessing.importance_pruning import get_feature_df
from bmiemg.preprocessing import TIME_FEATURE_FUNCTIONS, FREQ_FEATURE_FUNCTIONS

ARCHIVE_PATH = ROOT / "data/raw/emg_epn_612" / EPN612Specs.filename
ARCHIVE_NAME = Path(EPN612Specs.filename).stem
TRAINING_DIR = ARCHIVE_PATH.parent / ARCHIVE_NAME / ARCHIVE_NAME / "trainingJSON"


def _save_importance_plot(
    ranking: pd.DataFrame, key: str, path: Path, title: str | None = None
) -> None:
    """Draw a ranked bar plot in the style of the existing tree diagnostics."""
    fig, ax = plt.subplots(figsize=(9, max(4, 0.37 * len(ranking))))
    ranking.iloc[::-1].plot.barh(x=key, y="importance", ax=ax, legend=False)
    ax.set_xlabel("Impurity-based importance")
    ax.set_ylabel(key.replace("_", " ").title())
    ax.set_title(title or f"EPN612 8-channel {key.replace('_', ' ')} importance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _fit_and_evaluate(
    channels: Sequence[str],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    test_x: pd.DataFrame,
    test_y: np.ndarray,
    labels: list[str],
    output_dir: Path,
    metrics_filename: str,
    confusion_filename: str,
) -> tuple[dict, list]:
    """Fit and score the same six-tree baseline for any ordered channel set."""
    selected = validate_channels(channels)
    names = get_feature_names(selected)
    if list(train_x.columns) != names or list(test_x.columns) != names:
        raise ValueError("Feature columns differ from the selected canonical order")
    if set(train_y) != set(labels) or set(test_y) != set(labels):
        raise ValueError("Observed labels differ from the EPN612 label mapping")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    scores = np.empty((len(test_y), len(labels)), dtype=float)
    models = []
    for i, label in enumerate(labels):
        tree = DecisionTreeFactory(random_state=42).create()
        tree.fit(train_x, (train_y == label).astype(int))
        if list(tree.feature_names_in_) != names or len(
            tree.feature_importances_
        ) != len(names):
            raise ValueError(f"Feature alignment failed for {label}")
        positive = np.flatnonzero(tree.classes_ == 1)
        if len(positive) != 1:
            raise ValueError(f"No positive class for {label}")
        scores[:, i] = tree.predict_proba(test_x)[:, positive[0]]
        joblib.dump(tree, model_dir / f"{label}_vs_all.joblib")
        models.append(tree)
        print(f"Fitted {label} versus all for {selected}", flush=True)

    # One prediction per recording. Dataset label order breaks equal-score ties.
    predictions = np.asarray(labels)[scores.argmax(axis=1)]
    matrix = confusion_matrix(test_y, predictions, labels=labels)
    metrics = {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, predictions)),
        "macro_f1": float(
            f1_score(test_y, predictions, labels=labels, average="macro")
        ),
        "train_recordings": len(train_y),
        "test_recordings": len(test_y),
        "channels": list(selected),
        "feature_types": len(TIME_FEATURE_FUNCTIONS) + len(FREQ_FEATURE_FUNCTIONS),
        "model_inputs": len(names),
        "labels": labels,
        "evaluation": "trainingJSON/trainingSamples -> trainingJSON/testingSamples",
        "combination": "argmax of six positive-class predict_proba scores; label-code tie break",
    }
    (output_dir / metrics_filename).write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / confusion_filename, index_label="actual"
    )
    return metrics, models


def _save_reduced_importance(
    channels: tuple[str, ...],
    labels: list[str],
    models: list,
    output_dir: Path,
    key: str,
) -> pd.DataFrame:
    """Describe the fitted binary trees without changing the fixed channel set."""
    metadata = get_feature_df(
        SimpleNamespace(ch_names=channels),
        [*TIME_FEATURE_FUNCTIONS, *FREQ_FEATURE_FUNCTIONS],
    )
    names = get_feature_names(channels)
    expected = len(channels) * (
        len(TIME_FEATURE_FUNCTIONS) + len(FREQ_FEATURE_FUNCTIONS)
    )
    if len(metadata) != expected or metadata["feature"].tolist() != names:
        raise ValueError("Reduced-model feature metadata does not match input order")

    tree_rows = []
    for label, model in zip(labels, models, strict=True):
        if list(model.feature_names_in_) != names:
            raise ValueError(f"Saved binary tree input order differs for {label}")
        row = metadata.copy()
        row.insert(0, "tree_label", label)
        row["importance"] = model.feature_importances_
        if not np.isclose(row["importance"].sum(), 1.0):
            raise ValueError(f"Binary-tree importance does not sum to one: {label}")
        tree_rows.append(row)
    per_tree = pd.concat(tree_rows, ignore_index=True)
    if (
        set(per_tree["channel"]) != set(channels)
        or len(per_tree) != len(labels) * expected
    ):
        raise ValueError("A selected input is missing or a dropped channel is present")
    per_tree.to_csv(
        output_dir / f"per_tree_feature_channel_importance_{key}.csv", index=False
    )

    # Every tree has unit-sum impurity importance. Their mean gives one unit-sum
    # description of the six-tree collection without fitting a new ranker.
    mean_raw = metadata.copy()
    mean_raw["importance"] = np.mean(
        [model.feature_importances_ for model in models], axis=0
    )
    mean_raw.to_csv(
        output_dir / f"mean_raw_feature_channel_importance_{key}.csv", index=False
    )
    feature_rank = (
        mean_raw.groupby("feature_type", as_index=False, sort=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    canonical = {
        func.__name__ for func in [*TIME_FEATURE_FUNCTIONS, *FREQ_FEATURE_FUNCTIONS]
    }
    if (
        len(feature_rank) != len(canonical)
        or set(feature_rank["feature_type"]) != canonical
        or not (mean_raw.groupby("feature_type").size() == len(channels)).all()
        or not np.isclose(mean_raw["importance"].sum(), 1.0)
        or not np.isclose(feature_rank["importance"].sum(), 1.0)
    ):
        raise ValueError("Reduced-model feature importance aggregation failed")
    feature_rank.insert(0, "rank", np.arange(1, len(feature_rank) + 1))
    feature_rank.to_csv(output_dir / f"feature_importance_{key}.csv", index=False)
    _save_importance_plot(
        feature_rank,
        "feature_type",
        output_dir / f"feature_importance_{key}.png",
        f"EPN612 {key}: mean binary-tree feature-type importance",
    )
    return feature_rank
