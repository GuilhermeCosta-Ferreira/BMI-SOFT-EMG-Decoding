"""Evaluate cached high-pass windows while removing channels from eight to one."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import os

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache/matplotlib")
)

import argparse
import json
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from scripts.epn612_final import run_preprocessed_windowing as windowed
from scripts.epn612_final.run_importance_top_channels import channel_rank


def fit_remaining(
    channels,
    train_x,
    train_y,
    test_x,
    test_y,
    labels,
    output_dir,
    metrics_filename,
    confusion_filename,
):
    """Reuse valid individual trees from an interrupted subset fit."""
    names = windowed.analysis.get_feature_names(channels)
    factory = windowed.analysis.DecisionTreeFactory(random_state=42)
    if list(train_x.columns) != names or list(test_x.columns) != names:
        raise ValueError("Unexpected feature order")
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    scores = np.empty((len(test_y), len(labels)))
    models = []
    for i, label in enumerate(labels):
        path = model_dir / f"{label}_vs_all.joblib"
        tree = joblib.load(path) if path.is_file() else None
        if tree is not None and (
            tree.get_params() != factory.create().get_params()
            or list(tree.feature_names_in_) != names
            or not np.array_equal(tree.classes_, [0, 1])
        ):
            raise ValueError(f"Saved tree differs from this sweep: {path}")
        if tree is None:
            print(f"Training {len(channels)}ch / {label}", flush=True)
            tree = factory.create()
            tree.fit(train_x, (train_y == label).astype(int))
            joblib.dump(tree, path)
        else:
            print(f"Reusing {len(channels)}ch / {label}", flush=True)
        scores[:, i] = tree.predict_proba(test_x)[:, 1]
        models.append(tree)
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
        "channels": list(channels),
        "feature_types": 24,
        "model_inputs": len(names),
        "labels": labels,
        "evaluation": "trainingJSON/trainingSamples -> trainingJSON/testingSamples",
        "combination": "argmax of six positive-class predict_proba scores; label-code tie break",
    }
    (output_dir / metrics_filename).write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
        output_dir / confusion_filename, index_label="actual"
    )
    # Checkpoint graphs after each completed channel count.
    plot_results(output_dir.parent.parent)
    return metrics, models


def plot_results(output):
    """Plot measured points only; preserve gaps in an incomplete sweep."""
    ranking = tuple(
        pd.read_csv(output / "channel_pruning/channel_importance_8ch.csv")["channel"]
    )
    sweep = output / "channel_pruning"
    sweep.mkdir(exist_ok=True)
    rows, recalls = [], []
    labels = None
    supports = None
    for n in range(8, 0, -1):
        folder = sweep / f"top{n}"
        metrics_file = folder / f"metrics_top{n}.json"
        matrix_file = folder / f"confusion_matrix_top{n}.csv"
        if not metrics_file.is_file() or not matrix_file.is_file():
            continue
        metrics = json.loads(metrics_file.read_text())
        matrix = pd.read_csv(matrix_file, index_col="actual")
        values = matrix.to_numpy()
        if (
            metrics["channels"] != list(ranking[:n])
            or list(matrix.index) != metrics["labels"]
            or list(matrix.columns) != metrics["labels"]
            or values.sum() != metrics["test_recordings"]
            or not np.isclose(np.trace(values) / values.sum(), metrics["accuracy"])
        ):
            raise ValueError(f"Inconsistent saved results: {folder}")
        if labels is not None and (
            labels != metrics["labels"]
            or not np.array_equal(supports, values.sum(axis=1))
        ):
            raise ValueError("Channel counts use different held-out class supports")
        labels, supports = metrics["labels"], values.sum(axis=1)
        per_class = np.diag(values) / supports
        if not np.isclose(per_class.mean(), metrics["balanced_accuracy"]):
            raise ValueError("Balanced accuracy differs from confusion matrix")
        per_class_f1 = 2 * np.diag(values) / (supports + values.sum(axis=0))
        if not np.isclose(per_class_f1.mean(), metrics["macro_f1"]):
            raise ValueError("Macro F1 differs from confusion matrix")
        rows.append(
            {
                "n_channels": n,
                "n_dropped": 8 - n,
                "channels_used": "|".join(ranking[:n]),
                "channel_removed_at_this_step": ranking[n] if n < 8 else "none",
                **{
                    k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1")
                },
            }
        )
        for i, label in enumerate(labels):
            recalls.append(
                {
                    "n_channels": n,
                    "n_dropped": 8 - n,
                    "gesture": label,
                    "recall": float(per_class[i]),
                    "correct": int(values[i, i]),
                    "total_true": int(supports[i]),
                }
            )
    if not rows or rows[0]["n_channels"] != 8:
        raise ValueError("An evaluated eight-channel reference is required")
    results = pd.DataFrame(rows).set_index("n_dropped").reindex(range(8))
    complete = results["accuracy"].notna().all()
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        results[f"{metric}_drop_pp"] = -100 * results[metric].diff()
        results[f"{metric}_loss_vs_8ch_pp"] = 100 * (
            results.loc[0, metric] - results[metric]
        )
    measured = results.dropna(subset=["accuracy"]).reset_index()
    measured["n_channels"] = measured["n_channels"].astype(int)
    measured.to_csv(sweep / "channel_pruning_performance_changes.csv", index=False)
    recall = pd.DataFrame(recalls)
    recall.to_csv(sweep / "channel_pruning_recall.csv", index=False)
    wide = recall.pivot(index="gesture", columns="n_channels", values="recall")
    wide.reindex(index=labels, columns=range(8, 0, -1)).to_csv(
        sweep / "channel_pruning_recall_wide.csv"
    )
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    navy, muted = "#14263d", "#60758c"
    colors = ["#2965a3", "#0a8f82", "#df8342", "#7a57a8", "#cb5b78", "#697989"]
    x = np.arange(8)
    ticklabels = [f"{i}\n({8-i} ch)" for i in x]
    caption = (
        f"{int(supports.sum()):,} held-out 200 ms windows · fixed eight-channel ranking · "
        "six binary trees refitted per subset"
    )
    if not complete:
        caption += "\nIncomplete sweep: gaps indicate channel counts not yet evaluated."
    removal = "Removal order: " + " → ".join(ranking[:0:-1])

    def finish(fig, axes, name):
        for ax in np.atleast_1d(axes):
            ax.set_xticks(x, ticklabels)
            ax.set_xlim(-0.4, 7.4)
            ax.grid(axis="y", color="#dce4ed")
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("Channels removed (channels remaining)", color=navy)
        fig.text(0.07, 0.066, removal, color=muted, fontsize=10)
        fig.text(0.07, 0.025, caption, color=muted, fontsize=9)
        fig.tight_layout(rect=(0.015, 0.12, 0.985, 0.96))
        for extension in ("png", "svg"):
            fig.savefig(sweep / f"{name}.{extension}", dpi=200, facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 8))
    for metric, label, color in zip(
        ("accuracy", "balanced_accuracy", "macro_f1"),
        ("Accuracy", "Balanced accuracy", "Macro F1"),
        colors,
    ):
        ax.plot(x, results[metric], marker="o", linewidth=2.4, label=label, color=color)
    for i, value in results["accuracy"].dropna().items():
        ax.annotate(
            f"{value:.1%}",
            (i, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=navy,
        )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Held-out performance")
    ax.set_title(
        "EPN612 windowed: performance as channels are removed", color=navy, fontsize=19
    )
    ax.legend(frameon=False, loc="lower left")
    finish(fig, ax, "channel_pruning_accuracy")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    axes[0].set_title(
        "EPN612 windowed: performance lost per removed channel", color=navy, fontsize=19
    )
    for j, (metric, label, color) in enumerate(
        zip(
            ("accuracy", "balanced_accuracy", "macro_f1"),
            ("Accuracy", "Balanced accuracy", "Macro F1"),
            colors,
        )
    ):
        axes[0].bar(
            x + (j - 1) * 0.24,
            results[f"{metric}_drop_pp"],
            width=0.24,
            label=label,
            color=color,
        )
        axes[1].plot(
            x, results[f"{metric}_loss_vs_8ch_pp"], marker="o", color=color, label=label
        )
    for ax in axes:
        ax.axhline(0, color=muted, linewidth=1)
    axes[0].set_ylabel("Loss from previous step\n(percentage points)")
    axes[0].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.02))
    axes[0].margins(y=0.22)
    axes[1].set_title(
        "Positive loss = lower performance; negative loss = improvement",
        color=muted,
        fontsize=11,
    )
    axes[1].set_ylabel("Loss from eight-channel reference\n(percentage points)")
    finish(fig, axes, "channel_pruning_per_channel_drop")

    fig, ax = plt.subplots(figsize=(14, 8))
    for label, color in zip(labels, colors, strict=True):
        series = (
            recall.loc[recall.gesture == label]
            .set_index("n_dropped")["recall"]
            .reindex(x)
        )
        ax.plot(x, series, marker="o", linewidth=2, color=color, label=label)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Per-gesture recall")
    ax.set_title(
        "EPN612 windowed: gesture recall as channels are removed",
        color=navy,
        fontsize=19,
    )
    ax.legend(frameon=False, ncol=3, loc="lower left")
    finish(fig, ax, "channel_pruning_per_gesture_recall")
    print(f"Graphs updated: {len(rows)}/8 evaluated channel counts", flush=True)
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    if args.plot_only:
        plot_results(source)
        return
    config = json.loads((source / "config.json").read_text())
    expected_filters = {
        "notch_hz": 50,
        "notch_q": 30,
        "highpass_hz": 20,
        "highpass_order": 4,
        "zero_phase": True,
    }
    if (
        config["status"] != "complete"
        or config["filter_parameters"] != expected_filters
    ):
        raise ValueError("A completed 20 Hz high-pass + 50 Hz notch cache is required")
    if config["window_ms"] != 200 or config["step_ms"] != 100 or config["fs_hz"] != 200:
        raise ValueError("Unexpected cached sampling/window settings")
    labels = config.get("labels")
    if labels is None:
        labels = windowed.analysis.get_training_labels(
            windowed.analysis.TRAINING_DIR, windowed.analysis.ARCHIVE_PATH
        )
    extraction = config["extraction"]
    counts = extraction["windows"]
    full = np.load(source / "extracted_features/features.npy", mmap_mode="r")
    codes = np.load(source / "extracted_features/label_codes.npy", mmap_mode="r")
    if full.shape != (sum(counts.values()), len(windowed.NAMES)) or codes.shape != (
        len(full),
    ):
        raise ValueError("Cache shapes differ from the extraction report")
    if not np.isin(codes, np.arange(len(labels))).all():
        raise ValueError("Invalid cached label codes")
    ntrain = counts["trainingSamples"]
    train_y = np.asarray(labels)[codes[:ntrain]]
    test_y = np.asarray(labels)[codes[ntrain:]]
    for field, y in (("trainingSamples", train_y), ("testingSamples", test_y)):
        if pd.Series(y).value_counts().to_dict() != extraction["class_counts"][field]:
            raise ValueError("Cached label counts differ from the extraction report")
    ranking = channel_rank(
        pd.read_csv(source / "raw_feature_channel_importance_8ch.csv")
    )
    sweep = source / "channel_pruning"
    sweep.mkdir(exist_ok=True)
    ranking.to_csv(sweep / "channel_importance_8ch.csv", index=False)
    channels = tuple(ranking.channel)
    if len(channels) != 8 or set(channels) != set(windowed.EPN612_CHANNELS):
        raise ValueError("Expected all eight channels in the ranking")
    for size in range(8, 0, -1):
        selected = channels[:size]
        names = windowed.analysis.get_feature_names(selected)
        indices = [windowed.NAMES.index(name) for name in names]
        train_x = pd.DataFrame(
            np.asarray(full[:ntrain, indices], dtype=np.float32), columns=names
        )
        test_x = pd.DataFrame(
            np.asarray(full[ntrain:, indices], dtype=np.float32), columns=names
        )
        folder = sweep / f"top{size}"
        folder.mkdir(exist_ok=True)
        fit_remaining(
            selected,
            train_x,
            train_y,
            test_x,
            test_y,
            labels,
            folder,
            f"metrics_top{size}.json",
            f"confusion_matrix_top{size}.csv",
        )
        del train_x, test_x
    plot_results(source)
    (sweep / "completion.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "metric_unit": "held-out window",
                "evaluated_channel_counts": 8,
                "test_windows": len(test_y),
                "ranking": list(channels),
                "selection": "Descriptive ranking uses both labeled fields",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
