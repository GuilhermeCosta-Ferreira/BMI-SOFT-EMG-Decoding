# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar, Literal
from numpy.typing import NDArray
from .data_actor import DataActor

PartitionName = Literal["train", "val", "test"]


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DatasetPartition(DataActor):
    name: ClassVar[str] = "dataset_partition"
    partition_name: PartitionName
    signal: NDArray | None
    labels: NDArray | None
