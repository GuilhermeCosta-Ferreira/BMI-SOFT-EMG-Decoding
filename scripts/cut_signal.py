# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pathlib import Path

from bmiemg.data.convert import session_load, ChannelSplitter

from bmiemg.data.epoch import (
    SignalPartitioner,
    V1_TRIGGER_MAP,
    V2_TRIGGER_MAP,
)




# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data" / "bids"

FILE: Path = DATA / "sub-05/ses-02/sourcedata/sub-05_ses-02_task-Up_run-01_raw.xdf"



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    # 1. Load the session
    session = session_load(FILE)
    biotech_splitter = ChannelSplitter()
    bio_signal = biotech_splitter.split(session)
    signal = bio_signal.attach_annotations()
    emg_signal = signal["EMG"]

    # 2.
    movement_labels = sorted({
        desc for desc in emg_signal.annotations.description
        if str(desc).startswith("3")
    })
    print(movement_labels)


    events, event_id = mne.events_from_annotations(
        emg_signal,
        event_id={"32101": 32101},
    )
    """
    epochs = mne.Epochs(
        raw=emg_signal,
        events=events,
        event_id=event_id,
        tmin=-0.5,
        tmax=5.0,
        baseline=(-0.5, 0),
        preload=True,
        metadata=None,
    )
    """

    partinioner_v1 = SignalPartitioner(V1_TRIGGER_MAP)
    #partinioner_v2 = SignalPartitioner(V2_TRIGGER_MAP)

    epochs = partinioner_v1.partition(emg_signal)
    epochs = partinioner_v1.group(epochs)

    event_codes = epochs.events[:, 2]
    id_to_label = {v: k for k, v in epochs.event_id.items()}
    labels = [id_to_label[code] for code in event_codes]
    label_counts = pd.Series(labels).value_counts().sort_index()
    print("\nEpochs per label:")
    print(label_counts)
    print("\nTotal epochs:", len(epochs))

    epochs.plot(picks='all')
    plt.show()
