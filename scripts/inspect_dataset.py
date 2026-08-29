"""
This generates the model that is a triple decision tree, each one being
1 vs all.
"""
# ================================================================
# 0. Section: IMPORTS
# ================================================================
import os
import pickle
import mne
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import cast
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree, export_text


from bmiemg.postprocessing import prune, prune_features
from bmiemg.preprocessing.features import get_emg_features
from bmiemg.models.model_factories import DecisionTreeFactory
from bmiemg.models.target_builders import OneVsAllTargetBuilder
from bmiemg.preprocessing import (
    get_envelop,
    TIME_FEATURE_FUNCTIONS,
    FREQ_FEATURE_FUNCTIONS
)
from bmiemg.models import (
    TrainingConfig,
    ModelRegistry,
    Trainer,
    Evaluator
)



# ================================================================
# 1. Section: IMPORTS
# ================================================================
# 1.1. Paths
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data" / "bids"
DATASET_ROOT: Path = ROOT / "data" / "dataset"
DATASET: Path = DATASET_ROOT / "naive_archive_sensible_2026-05-16_epo.fif"
MODEL_SUFFIX: str = "_vs_all_6ch_scale_no_freq_no_envelop_sensible_details"
#vs_all_6ch_scale_no_freq_no_envelop_sensible

# 1.2 Preprocessing
WINDOW_SIZE: float = 0.1

# 1.3. Model
TRAIN_CONFIG: TrainingConfig = TrainingConfig(
    n_splits=5,
    random_state=42,
    scoring="accuracy",
    shuffle=True
)

TO_SAVE: bool = True


# ================================================================
# 1. Section: FUNCTIONS
# ================================================================




# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Loads the Dataset
    envelop_epochs = mne.read_epochs(DATASET, preload=True)
    event_codes = envelop_epochs.events[:, 2]

    id_to_label = {v: k for k, v in envelop_epochs.event_id.items()}

    labels = [id_to_label[code] for code in event_codes]

    label_counts = pd.Series(labels).value_counts().sort_index()

    print("\nEpochs per label:")
    print(label_counts)

    print("\nTotal epochs:", len(envelop_epochs))
