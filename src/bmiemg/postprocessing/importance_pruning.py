# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier


# ================================================================
# 1. Section: Functions
# ================================================================
def prune_features(
    epochs: mne.Epochs | mne.EpochsArray,
    features: np.ndarray,
    features_funcs: list,
    clf: DecisionTreeClassifier,
    nr_to_keep: int = 7,
) -> tuple:
    result, feature_names = prune_content(
        epochs=epochs,
        features=features,
        features_funcs=features_funcs,
        clf=clf,
        content="feature_type",
        nr_to_keep=nr_to_keep,
    )
    return result, feature_names


def prune_channels(
    epochs: mne.Epochs | mne.EpochsArray,
    features: np.ndarray,
    features_funcs: list,
    clf: DecisionTreeClassifier,
    nr_to_keep: int = 4,
) -> np.ndarray:
    _, channels = prune_content(
        epochs=epochs,
        features=features,
        features_funcs=features_funcs,
        clf=clf,
        content="channel",
        nr_to_keep=nr_to_keep,
    )

    return np.asarray(channels)


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def get_feature_df(
    epochs: mne.Epochs | mne.EpochsArray,
    features_funcs: list,
) -> pd.DataFrame:
    rows = []

    for feature_func in features_funcs:
        for ch_name in epochs.ch_names:
            rows.append(
                {
                    "feature_type": feature_func.__name__,
                    "channel": ch_name,
                    "feature": f"{feature_func.__name__}_{ch_name}",
                }
            )

    return pd.DataFrame(rows)


def prune_content(
    epochs: mne.Epochs | mne.EpochsArray,
    features: np.ndarray,
    features_funcs: list,
    clf: DecisionTreeClassifier,
    content: str = "feature_type",
    nr_to_keep: int = 7,
):
    feature_info = get_feature_df(epochs, features_funcs)
    feature_info["importance"] = clf.feature_importances_

    feature_type_importance = (
        feature_info.groupby(content, as_index=False)["importance"]  # type: ignore
        .sum()
        .sort_values("importance", ascending=False)
    )

    top_feature_types = feature_type_importance.head(nr_to_keep)[content].tolist()

    print(f"Selected: {top_feature_types}")

    selected_indices = feature_info.index[
        feature_info[content].isin(top_feature_types)
    ].to_numpy()

    return features[:, selected_indices], top_feature_types


def prune(
    epochs: mne.Epochs | mne.EpochsArray,
    features_funcs: list,
    clf: DecisionTreeClassifier,
    top_feature_types: list,
    content: str = "feature_type",
):
    feature_info = get_feature_df(epochs, features_funcs)
    feature_info["importance"] = clf.feature_importances_

    selected_indices = feature_info.index[
        feature_info[content].isin(top_feature_types)
    ].to_numpy()

    return selected_indices
