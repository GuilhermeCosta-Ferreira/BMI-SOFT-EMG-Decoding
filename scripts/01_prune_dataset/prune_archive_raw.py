# ================================================================
# 0. Section: IMPORTS
# ================================================================
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from bmiemg.prune import DatasetPruner
from bmiemg.prune.domain.actors import Dataset, DatasetPartition, TrialData


# ================================================================
# 1. Section: INPUTS
# ================================================================
SOURCE_NAME: list[str] = [
    "bids",
]

_PARTITIONS = ["train", "val", "test"]


# ================================================================
# 2. Section: PLOTS
# ================================================================
def plot_dataset_balance(dataset: Dataset) -> None:
    client = {p.partition_name: p for p in dataset.client}
    server = {p.partition_name: p for p in dataset.server}
    # Test markers are blanked on the client, so read them from the server.
    label_source = {**client, "test": server["test"]}

    _, axes = plt.subplots(1, 3, figsize=(16, 4))
    _plot_sizes(axes[0], client)
    _plot_balance(axes[1], _subject_counts(client), "user balance (subject)")
    _plot_balance(axes[2], _movement_counts(label_source), "movement balance (code)")

    plt.suptitle("TTV dataset balance")
    plt.tight_layout()
    plt.show()


def _plot_sizes(ax, partitions: dict[str, DatasetPartition]) -> None:
    sizes = [len(partitions[p].trials) for p in _PARTITIONS]
    ax.bar(_PARTITIONS, sizes, color=["C0", "C1", "C2"])
    for x, n in enumerate(sizes):
        ax.text(x, n, str(n), ha="center", va="bottom")
    ax.set_title("partition sizes")
    ax.set_ylabel("samples")


def _plot_balance(ax, counts: dict, title: str) -> None:
    """Stacked bar of a category's distribution across the partitions."""
    categories = sorted({c for part in counts.values() for c in part})
    bottom = np.zeros(len(_PARTITIONS))
    for category in categories:
        heights = np.array(
            [counts.get(p, Counter()).get(category, 0) for p in _PARTITIONS]
        )
        ax.bar(_PARTITIONS, heights, bottom=bottom, label=str(category))
        bottom += heights
    ax.set_title(title)
    ax.set_ylabel("samples")
    ax.legend(fontsize="x-small", ncol=2)


def _subject_counts(partitions: dict[str, DatasetPartition]) -> dict:
    return {
        name: Counter(_subject(t) for t in part.trials)
        for name, part in partitions.items()
    }


def _movement_counts(partitions: dict[str, DatasetPartition]) -> dict:
    return {
        name: Counter(_dominant_movement(t.markers) for t in part.trials)
        for name, part in partitions.items()
    }


def _subject(trial: TrialData) -> int:
    return -1 if trial.subject_number is None else trial.subject_number


def _dominant_movement(markers: np.ndarray) -> int:
    values = markers[markers != 0]
    movements = values % 100
    movements = movements[movements != 0]
    if movements.size == 0:
        return 0
    codes, counts = np.unique(movements, return_counts=True)
    return int(codes[np.argmax(counts)])


# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    pruner = DatasetPruner(
        source_name=SOURCE_NAME,
    )

    actors = pruner.run("raw_archive_v1")

    if actors and isinstance(actors[0], Dataset):
        plot_dataset_balance(actors[0])
