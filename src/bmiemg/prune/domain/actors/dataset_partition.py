# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from mne.io import BaseRaw
from dataclasses import dataclass
from typing import ClassVar, Literal, cast

from .data_actor import DataActor
from .trial_data import TrialData

PartitionName = Literal["train", "val", "test"]
_STIM_CH_NAME = "STI 014"



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DatasetPartition(DataActor):
    name: ClassVar[str] = "dataset_partition"
    partition_name: PartitionName
    trials: list[TrialData]

    @property
    def mne_dataset(self) -> BaseRaw:
        if not self.trials:
            raise ValueError(
                f"partition {self.partition_name!r} has no trials to build a Raw from"
            )

        raws = [_trial_to_raw(trial) for trial in self.trials]
        return cast(BaseRaw, mne.concatenate_raws(raws))


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Module Helpers
# ──────────────────────────────────────────────────────
def _trial_to_raw(trial: TrialData) -> mne.io.RawArray:
    if trial.signal_type is None:
        raise ValueError(
            f"trial {trial.file_name!r} has no signal_type; cannot type its channels"
        )

    # Stack the per-sample markers under the signal as a dedicated stim channel.
    data = np.vstack([trial.signal_ch, trial.markers])
    ch_names = [*trial.channel_names, _STIM_CH_NAME]

    # Type every channel as the signal modality, then retag the markers row.
    info = mne.create_info(
        ch_names=ch_names, sfreq=trial.sfreq, ch_types=trial.signal_type
    )
    raw = mne.io.RawArray(data, info)
    raw.set_channel_types({_STIM_CH_NAME: "stim"})
    return raw
