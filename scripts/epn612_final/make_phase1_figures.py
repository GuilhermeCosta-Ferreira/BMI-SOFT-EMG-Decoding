"""Presentation figures for final importance and optional eight-channel evaluation."""

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
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

NAVY = "#14263d"
BLUE = "#2965a3"
TEAL = "#0a8f82"
ORANGE = "#df8342"
MUTED = "#60758c"
GRID = "#dce4ed"
PALE = "#f4f8fb"


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.png", dpi=200, facecolor="white")
    fig.savefig(OUTPUT / f"{name}.svg", facecolor="white")
    plt.close(fig)


def _base(title: str, subtitle: str) -> plt.Figure:
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.055, 0.94, title, fontsize=27, fontweight="bold", color=NAVY)
    fig.text(0.055, 0.895, subtitle, fontsize=13, color=MUTED)
    return fig


def _barh(
    ax: plt.Axes,
    names: list[str],
    values: np.ndarray,
    colors: list[str],
    xmax: float,
    fontsize: float = 12,
) -> None:
    xmax = max(xmax, float(np.max(values)) * 1.18)
    y = np.arange(len(names))
    ax.barh(y, values, color=colors, height=0.70)
    ax.set_yticks(y, names, fontsize=fontsize, color=NAVY)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0, pad=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for yi, value in zip(y, values):
        ax.text(
            value + xmax * 0.012,
            yi,
            f"{value:.1%}",
            va="center",
            fontsize=fontsize - 1,
            color=NAVY,
            fontweight="semibold",
        )


def make_summary(metrics: dict, channels: pd.DataFrame, features: pd.DataFrame) -> None:
    fig = _base(
        "EPN612 | Full-channel Phase 1",
        "Eight-channel TreeOfTrees baseline and global input rankings",
    )
    cards = [
        ("ACCURACY", metrics["accuracy"], BLUE),
        ("BALANCED ACCURACY", metrics["balanced_accuracy"], TEAL),
        ("MACRO F1", metrics["macro_f1"], ORANGE),
    ]
    for i, (label, value, color) in enumerate(cards):
        left = 0.055 + i * 0.305
        box = plt.Rectangle(
            (left, 0.675), 0.275, 0.15, transform=fig.transFigure, color=PALE, zorder=0
        )
        fig.add_artist(box)
        fig.text(
            left + 0.017, 0.78, label, fontsize=12, color=MUTED, fontweight="semibold"
        )
        fig.text(
            left + 0.017,
            0.698,
            f"{value:.1%}",
            fontsize=32,
            fontweight="bold",
            color=color,
        )

    ax_ch = fig.add_axes((0.09, 0.17, 0.37, 0.40))
    ax_ft = fig.add_axes((0.59, 0.17, 0.32, 0.40))
    fig.text(
        0.055, 0.605, "CHANNEL IMPORTANCE", fontsize=15, fontweight="bold", color=NAVY
    )
    fig.text(
        0.555, 0.605, "TOP FEATURE TYPES", fontsize=15, fontweight="bold", color=NAVY
    )
    _barh(
        ax_ch,
        channels.channel.tolist(),
        channels.importance.to_numpy(),
        [TEAL] + [BLUE] * (len(channels) - 1),
        0.37,
        fontsize=12,
    )
    top = features.head(6)
    _barh(
        ax_ft,
        top.feature_type.tolist(),
        top.importance.to_numpy(),
        [ORANGE] + [BLUE] * (len(top) - 1),
        0.57,
        fontsize=12,
    )
    fig.text(
        0.055,
        0.075,
        "Performance: held-out annotated windows. Importance: separate multiclass ranking tree on both labeled fields.",
        fontsize=11,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.043,
        "Importance is a share of tree impurity reduction; it does not measure the effect of removing an input.",
        fontsize=10,
        color=MUTED,
    )
    _save(fig, "phase1_summary")


def make_channel_rank(channels: pd.DataFrame) -> None:
    fig = _base(
        "Which EMG channels drive the ranking tree?",
        "Importance summed across all 24 canonical feature types per channel",
    )
    ax = fig.add_axes((0.18, 0.14, 0.72, 0.64))
    _barh(
        ax,
        channels.channel.tolist(),
        channels.importance.to_numpy(),
        [TEAL] + [BLUE] * (len(channels) - 1),
        0.36,
        fontsize=18,
    )
    ax.set_xlabel(
        "Share of total impurity-based importance", fontsize=13, color=NAVY, labelpad=14
    )
    fig.text(
        0.055,
        0.055,
        "Source: general multiclass importance tree on filtered annotated windows from both labeled fields.",
        fontsize=11,
        color=MUTED,
    )
    _save(fig, "channel_importance_presentation")


def make_feature_rank(features: pd.DataFrame) -> None:
    fig = _base(
        "Which EMG feature types drive the ranking tree?",
        "Importance summed across eight channels for each of 24 canonical features",
    )
    ax = fig.add_axes((0.24, 0.14, 0.67, 0.66))
    _barh(
        ax,
        features.feature_type.tolist(),
        features.importance.to_numpy(),
        [ORANGE, TEAL] + [BLUE] * (len(features) - 2),
        0.56,
        fontsize=10.5,
    )
    ax.set_xlabel(
        "Share of total impurity-based importance", fontsize=12, color=NAVY, labelpad=10
    )
    fig.text(
        0.055,
        0.055,
        "50 Hz notch + 20 Hz high-pass; importance summed over all eight channels.",
        fontsize=11,
        color=MUTED,
    )
    _save(fig, "feature_importance_presentation")


def make_confusion(metrics: dict, confusion: pd.DataFrame) -> None:
    fig = _base(
        "Where does the eight-channel baseline confuse gestures?",
        f"Held-out window-level predictions | {metrics['accuracy']:.1%} overall accuracy",
    )
    values = confusion.to_numpy(dtype=int)
    row_totals = values.sum(axis=1)
    rates = values / row_totals[:, None]
    ax = fig.add_axes((0.18, 0.18, 0.62, 0.62))
    image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(np.arange(6), confusion.columns, fontsize=12, color=NAVY)
    ax.set_yticks(np.arange(6), confusion.index, fontsize=12, color=NAVY)
    ax.set_xlabel("Predicted gesture", fontsize=13, color=NAVY, labelpad=13)
    ax.set_ylabel("Actual gesture", fontsize=13, color=NAVY, labelpad=13)
    ax.tick_params(length=0)
    for i in range(6):
        for j in range(6):
            ax.text(
                j,
                i,
                f"{rates[i, j]:.1%}\n{values[i, j]:,}",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if rates[i, j] > 0.52 else NAVY,
                fontweight="semibold" if i == j else "normal",
            )
    cbar = fig.colorbar(image, ax=ax, shrink=0.76, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    cbar.set_label("Percent of actual class", fontsize=11, color=NAVY)
    fig.text(
        0.055,
        0.055,
        "Rows contain the retained testingSamples windows per gesture. Trees fit trainingSamples windows.",
        fontsize=11,
        color=MUTED,
    )
    _save(fig, "confusion_matrix_presentation")


def main():
    import argparse
    from scripts.epn612_final.run_importance_top_channels import channel_rank

    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    OUTPUT = source / "showcase"
    channels = channel_rank(
        pd.read_csv(source / "raw_feature_channel_importance_8ch.csv")
    )
    features = pd.read_csv(source / "feature_importance_8ch.csv")
    make_channel_rank(channels)
    make_feature_rank(features)
    evaluated = source / "channel_pruning/top8"
    if (evaluated / "metrics_top8.json").is_file():
        metrics = json.loads((evaluated / "metrics_top8.json").read_text())
        confusion = pd.read_csv(
            evaluated / "confusion_matrix_top8.csv", index_col="actual"
        )
        make_summary(metrics, channels, features)
        make_confusion(metrics, confusion)
    print(f"Figures saved in {OUTPUT}")


if __name__ == "__main__":
    main()
