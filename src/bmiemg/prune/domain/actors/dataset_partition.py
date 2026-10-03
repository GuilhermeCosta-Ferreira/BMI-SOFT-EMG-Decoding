# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar, Literal

from .data_actor import DataActor
from .trial_data import TrialData

PartitionName = Literal["train", "val", "test"]


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DatasetPartition(DataActor):
    name: ClassVar[str] = "dataset_partition"
    partition_name: PartitionName
    trials: list[TrialData]
