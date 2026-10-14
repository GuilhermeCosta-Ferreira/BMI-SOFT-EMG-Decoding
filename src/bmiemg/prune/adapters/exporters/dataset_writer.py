# ================================================================
# 0. Section: IMPORTS
# ================================================================
import shutil

from pathlib import Path

from .writer import Writer
from .export_context import ExportContext
from ...domain.actors import Dataset, DatasetPartition
from .writer_registry import WriterRegistry


# ================================================================
# 1. Section: Class definition
# ================================================================
@WriterRegistry.register(Dataset.name)
class DatasetWriter(Writer[Dataset]):
    def write(self, actor: Dataset, dest_dir: Path, ctx: ExportContext) -> None:
        for group, partitions in (("client", actor.client), ("server", actor.server)):
            group_dir = dest_dir / group
            for partition in partitions:
                self._write_partition(partition, group_dir, ctx)

            # Only zip a group that actually produced files.
            if ctx.zip_groups and group_dir.is_dir():
                shutil.make_archive(str(group_dir), "zip", root_dir=group_dir)

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _write_partition(
        self, partition: DatasetPartition, dest_dir: Path, ctx: ExportContext
    ) -> None:
        if not partition.trials:
            print(f"export: skipping empty partition {partition.partition_name!r}")
            return

        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            dest_dir / f"{ctx.protocol_name}_{partition.partition_name}_{ctx.timestamp}"
        )
        match ctx.fmt:
            case ".fif":
                self._write_fif(partition, stem)
            case _:
                raise ValueError(f"DatasetWriter cannot export format {ctx.fmt!r}")

    def _write_fif(self, partition: DatasetPartition, stem: Path) -> None:
        # mne requires a Raw filename ending in raw.fif to avoid a warning.
        target = stem.with_name(f"{stem.name}_raw.fif")
        partition.mne_dataset.save(target, overwrite=True)
