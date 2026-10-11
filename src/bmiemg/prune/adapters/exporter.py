# ================================================================
# 0. Section: IMPORTS
# ================================================================
import shutil

from pathlib import Path
from dataclasses import dataclass
from datetime import UTC, datetime

from .source import Source
from ..domain import DataActor
from .exporters import Writer, WriterRegistry
from ..domain.actors import Dataset, DatasetPartition



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Exporter:
    source: Source

    def export(
        self,
        actors: list[DataActor],
        protocol_name: str,
        fmt: str,
        zip_groups: bool,
    ) -> None:
        # 1. Get the needed resources for the writing
        writer = WriterRegistry.get(fmt)
        out_dir = self.source.out_dir(protocol_name)
        timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S")

        # 2. Loops over each actor
        for actor in actors:
            if isinstance(actor, Dataset):
                self._export_group(
                    actor,
                    out_dir,
                    protocol_name,
                    timestamp,
                    writer,
                    zip_groups,
                    )
            elif isinstance(actor, DatasetPartition):
                self._export_partition(actor, out_dir, protocol_name, timestamp, writer)
            else:
                raise TypeError(
                    f"export expects Dataset or DatasetPartition, got "
                    f"{type(actor).__name__}"
                )


    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _export_group(
        self,
        actor: Dataset,
        out_dir: Path,
        protocol_name: str,
        timestamp: str,
        writer: Writer,
        zip_group: bool,
    ) -> None:
        for group, partitions in (
            ("client", actor.client),
            ("server", actor.server),
        ):
            group_dir = out_dir / group
            for partition in partitions:
                self._export_partition(
                    partition, group_dir, protocol_name, timestamp, writer
                )

            # Only zip a group that actually produced files.
            if zip_group and group_dir.is_dir():
                shutil.make_archive(str(group_dir), "zip", root_dir=group_dir)

    def _export_partition(
        self,
        partition: DatasetPartition,
        dest_dir: Path,
        protocol_name: str,
        timestamp: str,
        writer: Writer,
    ) -> None:
        if not partition.trials:
            print(f"export: skipping empty partition {partition.partition_name!r}")
            return

        dest_dir.mkdir(parents=True, exist_ok=True)
        stem = dest_dir / f"{protocol_name}_{partition.partition_name}_{timestamp}"
        writer.write(partition, stem)
