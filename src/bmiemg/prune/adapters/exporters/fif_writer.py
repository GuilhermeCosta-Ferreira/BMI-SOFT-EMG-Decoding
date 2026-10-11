# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from typing import ClassVar

from .writer import Writer
from ...domain.actors import DatasetPartition
from .writer_registry import WriterRegistry


# ================================================================
# 1. Section: Class definition
# ================================================================
@WriterRegistry.register(".fif")
class FifWriter(Writer):
    suffix: ClassVar[str] = ".fif"

    def write(self, partition: DatasetPartition, dest_stem: Path) -> None:
        target = dest_stem.with_name(f"{dest_stem.name}_raw{self.suffix}")
        partition.mne_dataset.save(target, overwrite=True)
