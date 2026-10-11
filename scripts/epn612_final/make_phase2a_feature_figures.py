"""Draw final per-gesture and global feature-type importance plots."""

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
MUTED = "#60758c"
GRID = "#dce4ed"


def _plot_feature_types(
    ranking: pd.DataFrame,
    title: str,
    subtitle: str,
    note: str,
    path: Path,
    svg: bool = False,
) -> None:
    """Make a readable 16:9 chart with exactly one bar per feature type."""
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    fig.text(0.055, 0.94, title, fontsize=25, fontweight="bold", color=NAVY)
    fig.text(0.055, 0.895, subtitle, fontsize=12, color=MUTED)
    ax = fig.add_axes((0.24, 0.145, 0.67, 0.67))
    values = ranking["importance"].to_numpy()
    names = ranking["feature_type"].tolist()
    y = np.arange(len(names))
    colors = [TEAL] + [BLUE] * (len(names) - 1)
    ax.barh(y, values, height=0.72, color=colors)
    ax.set_yticks(y, names, fontsize=10, color=NAVY)
    ax.invert_yaxis()
    xmax = max(0.57, float(values.max()) * 1.12)
    ax.set_xlim(0, xmax)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(
        "Share of total impurity-based importance", fontsize=12, color=NAVY, labelpad=11
    )
    ax.tick_params(axis="x", colors=MUTED, labelsize=10, length=0)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, value in enumerate(values):
        ax.text(
            value + xmax * 0.008,
            i,
            f"{value:.1%}",
            va="center",
            fontsize=9,
            color=NAVY,
            fontweight="semibold",
        )
    fig.text(0.055, 0.055, note, fontsize=10, color=MUTED)
    fig.savefig(path, dpi=200, facecolor="white")
    if svg:
        fig.savefig(path.with_suffix(".svg"), facecolor="white")
    plt.close(fig)
