# ================================================================
# 0. Section: IMPORTS
# ================================================================
import shutil

from pathlib import Path
from typing import ClassVar
from dataclasses import dataclass

from .writer import Writer
from ...domain.actors import Dataset, DatasetPartition
from .writer_registry import WriterRegistry


# ================================================================
# 1. Section: Class definition
# ================================================================
@WriterRegistry.register("dataset")
@dataclass
class DatasetWriter(Writer[Dataset]):
    name: ClassVar[str] = "dataset"
    actor: ClassVar[str] = Dataset.name
    fmt: str = ".fif"
    zip_groups: bool = True

    def write(self, actor: Dataset, dest_dir: Path) -> None:
        for group, partitions in (("client", actor.client), ("server", actor.server)):
            group_dir = dest_dir / group
            for partition in partitions:
                self._write_partition(partition, group_dir)

            # Only zip a group that actually produced files.
            if self.zip_groups and group_dir.is_dir():
                shutil.make_archive(str(group_dir), "zip", root_dir=group_dir)

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _write_partition(self, partition: DatasetPartition, dest_dir: Path) -> None:
        if not partition.trials:
            print(f"export: skipping empty partition {partition.partition_name!r}")
            return

        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = dest_dir / f"{partition.partition_name}_{self.timestamp}"
        match self.fmt:
            case ".fif":
                self._write_fif(partition, stem)
            case _:
                raise ValueError(f"DatasetWriter cannot export format {self.fmt!r}")

    def _write_fif(self, partition: DatasetPartition, stem: Path) -> None:
        # mne requires a Raw filename ending in raw.fif to avoid a warning.
        target = stem.with_name(f"{stem.name}_raw.fif")
        partition.mne_dataset.save(target, overwrite=True)
