# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..actors import Dataset, DatasetPartition, TrialData
from .prune_step import PruneStep
from .ttv_split_specs import TTVSplitSpecs


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class TTVSplitStep(PruneStep[TTVSplitSpecs, TrialData, Dataset]):
    name: ClassVar[str] = "ttv_split"
    config: TTVSplitSpecs

    def apply(self, actors: list[TrialData]) -> list[Dataset]:
        rng = np.random.default_rng(self.config.seed)

        train: list[TrialData] = []
        val: list[TrialData] = []
        test: list[TrialData] = []
        # Split each stratum on its own so the requested balances hold.
        for stratum in self._strata(actors).values():
            t, v, s = self._split_stratum(stratum, rng)
            train += t
            val += v
            test += s

        # The dev-facing side only sees the test signal; the true test labels
        # (markers) stay on the server.
        test_unlabelled = [_hide_markers(trial) for trial in test]
        client = [
            DatasetPartition(partition_name="train", trials=train),
            DatasetPartition(partition_name="val", trials=val),
            DatasetPartition(partition_name="test", trials=test_unlabelled),
        ]
        server = [DatasetPartition(partition_name="test", trials=test)]

        return [Dataset(client=client, server=server)]

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _strata(self, actors: list[TrialData]) -> dict[tuple, list[TrialData]]:
        strata: dict[tuple, list[TrialData]] = {}
        for actor in actors:
            strata.setdefault(self._stratum_key(actor), []).append(actor)
        return strata

    def _stratum_key(self, actor: TrialData) -> tuple:
        key: list[object] = []
        if self.config.user_balance:
            key.append(actor.subject_number)
        if self.config.movement_balance:
            key.append(_dominant_movement(actor.markers))
        return tuple(key)

    def _split_stratum(
        self, actors: list[TrialData], rng: np.random.Generator
    ) -> tuple[list[TrialData], list[TrialData], list[TrialData]]:
        shuffled = [actors[i] for i in rng.permutation(len(actors))]

        n = len(shuffled)
        n_train = min(round(n * self.config.train_ratio), n)
        n_val = min(round(n * self.config.val_ratio), n - n_train)

        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]
        return train, val, test


# ──────────────────────────────────────────────────────
# 1.2 Subsection: Module Helpers
# ──────────────────────────────────────────────────────
def _hide_markers(trial: TrialData) -> TrialData:
    """Blank the markers so the client never sees the test labels."""
    return trial.copy_with(markers=np.zeros_like(trial.markers))


def _dominant_movement(markers: NDArray) -> int:
    values = markers[markers != 0]
    movements = values % 100
    movements = movements[movements != 0]
    if movements.size == 0:
        return 0
    codes, counts = np.unique(movements, return_counts=True)
    return int(codes[np.argmax(counts)])
