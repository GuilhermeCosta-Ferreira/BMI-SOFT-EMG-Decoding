# ================================================================
# 0. Section: IMPORTS
# ================================================================
import shutil

from pathlib import Path
from typing import ClassVar
from dataclasses import dataclass
from datetime import UTC, datetime

from .prune_step import PruneStep
from ..actors import DataActor, Dataset, DatasetPartition


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class SaveDatasetStep(PruneStep[DataActor, DataActor]):
    name: ClassVar[str] = "save_dataset"
    protocol_name: str
    output_dir: Path
    overwrite: bool = True
    zip_folders: bool = False

    def apply(self, actors: list[DataActor]) -> list[DataActor]:
        # One timestamp per run so a dataset's files share a suffix.
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")

        for actor in actors:
            if isinstance(actor, Dataset):
                self._save_group(actor.client, "client", timestamp)
                self._save_group(actor.server, "server", timestamp)
            elif isinstance(actor, DatasetPartition):
                self._save_partition(actor, self.output_dir, timestamp)
            else:
                raise TypeError(
                    f"save_dataset expects Dataset or DatasetPartition, got "
                    f"{type(actor).__name__}"
                )

        # Side-effect step: the actors continue unchanged down the pipeline.
        return actors

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _save_group(
        self, partitions: list[DatasetPartition], group: str, timestamp: str
    ) -> None:
        group_dir = self.output_dir / group
        for partition in partitions:
            self._save_partition(partition, group_dir, timestamp)

        # Only zip a group that actually produced files.
        if self.zip_folders and group_dir.is_dir():
            shutil.make_archive(str(group_dir), "zip", root_dir=group_dir)

    def _save_partition(
        self, partition: DatasetPartition, dest_dir: Path, timestamp: str
    ) -> None:
        if not partition.trials:
            print(f"save_dataset: skipping empty partition {partition.partition_name!r}")
            return

        dest_dir.mkdir(parents=True, exist_ok=True)
        # mne requires a Raw filename ending in `raw.fif` to avoid a warning.
        name = f"{self.protocol_name}_{partition.partition_name}_{timestamp}"
        partition.mne_dataset.save(dest_dir / f"{name}_raw.fif", overwrite=self.overwrite)
