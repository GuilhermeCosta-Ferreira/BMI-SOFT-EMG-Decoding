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
ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data" / "bids"
DATASET_ROOT: Path = ROOT / "data" / "dataset"
DATASET: Path = DATASET_ROOT / "naive_archive_sensible_2026-05-16_epo.fif"
MODEL_SUFFIX: str = "_vs_all_5ch_scale_no_freq_no_envelop_sensible"
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
def save_tree_diagnostics(
    clf: DecisionTreeClassifier,
    feature_names: list[str],
    model_name: str,
    mean_score: float,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    params = clf.get_params()

    stats = {
        "model": model_name,
        "mean_accuracy": mean_score,
        "depth": clf.get_depth(),
        "n_leaves": clf.get_n_leaves(),
        "n_nodes": clf.tree_.node_count,
        "n_features_input": clf.n_features_in_,
        "n_features_used": int(np.sum(clf.feature_importances_ > 0)),
        "criterion": params.get("criterion"),
        "splitter": params.get("splitter"),
        "max_depth_param": params.get("max_depth"),
        "min_samples_split": params.get("min_samples_split"),
        "min_samples_leaf": params.get("min_samples_leaf"),
        "ccp_alpha": params.get("ccp_alpha"),
    }

    # ------------------------------------------------------------
    # 1. Feature importance plot
    # ------------------------------------------------------------
    importances = pd.Series(
        clf.feature_importances_,
        index=feature_names,
    ).sort_values()

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(importances))))
    importances.plot(kind="barh", ax=ax)
    ax.set_title(f"{model_name}: feature importance")
    ax.set_xlabel("Impurity-based importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_feature_importance.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------
    # 2. Tree plot, limited depth so it stays readable
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        max_depth=3,
        ax=ax,
    )
    ax.set_title(f"{model_name}: decision tree, first 3 levels")
    fig.tight_layout()
    fig.savefig(out_dir / f"{model_name}_tree_depth3.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------
    # 3. Text version of the tree rules
    # ------------------------------------------------------------
    tree_text = export_text(
        clf,
        feature_names=feature_names,
        max_depth=5,
    )

    with open(out_dir / f"{model_name}_tree_rules.txt", "w") as f:
        f.write(tree_text)

    # ------------------------------------------------------------
    # 4. Save stats
    # ------------------------------------------------------------
    pd.DataFrame([stats]).to_csv(
        out_dir / f"{model_name}_tree_stats.csv",
        index=False,
    )

    return stats

def expand_pruned_feature_names(
    epochs: mne.Epochs,
    selected_feature_names: list[str],
    selected_features: np.ndarray,
    order: str = "feature_major",
) -> list[str]:
    """
    Expands feature names from:

        ['log_det', 'mav', ...]

    to channel-specific names:

        ['log_det_AUX7', 'log_det_AUX12', ...]

    The final length must match selected_features.shape[1].
    """

    ch_names = list(epochs.ch_names)
    n_cols = selected_features.shape[1]

    # Already correct
    if len(selected_feature_names) == n_cols:
        return selected_feature_names

    # Common case: 7 feature types × 6 channels = 42 columns
    if len(selected_feature_names) * len(ch_names) == n_cols:
        if order == "feature_major":
            return [
                f"{feature}_{ch}"
                for feature in selected_feature_names
                for ch in ch_names
            ]

        if order == "channel_major":
            return [
                f"{ch}_{feature}"
                for ch in ch_names
                for feature in selected_feature_names
            ]

        raise ValueError(f"Unknown order: {order}")

    # Safe fallback
    return [f"x{i}" for i in range(n_cols)]




# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # 1. Loads the Dataset
    envelop_epochs = mne.read_epochs(DATASET, preload=True)

    # 2. Extract the envelop for only 4 channels
    #envelop_epochs = get_envelop(
    #    cast(mne.EpochsArray, epochs),
    #    window_s=WINDOW_SIZE,
    #)
    envelop_epochs = envelop_epochs.copy().pick(['AUX7', 'AUX12', 'AUX8', 'AUX11', 'AUX10'])
    #envelop_epochs = envelop_epochs.copy().pick(['AUX12'])

    # 3. Extract the features
    features = get_emg_features(
        envelop_epochs.get_data(),
        envelop_epochs.info["sfreq"],
        TIME_FEATURE_FUNCTIONS,
        FREQ_FEATURE_FUNCTIONS
    )
    print(
        f"Extracted {features.shape[1]} features from "
        f"{features.shape[0]} epochs\n"
    )

    # 4. Get all label names (we want to build one model per label)
    models = list(envelop_epochs.event_id.keys())
    print(f"Building the follwing models: {models}")

    # 5. Initializes the model architecture
    registry = ModelRegistry()
    registry.register(
        "dt",
        DecisionTreeFactory(
            random_state=42,
        ),
    )
    trainer = Trainer(
        registry=registry,
        target_builder=OneVsAllTargetBuilder(),
        evaluator=Evaluator(TRAIN_CONFIG),
    )

    # 5. Builds each model
    best_features = []
    for model in models:
        print(f"|----- {model.title()} Model Training -----|")
        result = trainer.train(
            epochs=envelop_epochs,
            features=features,
            model_name="dt",
            target_name=model,
        )
        clf = result.estimator

        selected_indices = prune(
            envelop_epochs,
            list(np.concat([TIME_FEATURE_FUNCTIONS, FREQ_FEATURE_FUNCTIONS], axis=0)),
            cast(DecisionTreeClassifier, clf),
            ['log_det', 'mav', 'maxav', 'rms', 'ssc', 'std', 'wl'],
        )
        selected_features = features[:, selected_indices]
        """
        selected_features, feature_names = prune_features(
            envelop_epochs,
            features,
            list(np.concat([TIME_FEATURE_FUNCTIONS, FREQ_FEATURE_FUNCTIONS], axis=0)),
            cast(DecisionTreeClassifier, clf),
        )
        best_features.append(feature_names)
        print(f"Feature names {feature_names}")
        """
        print("Original:", features.shape)
        print("Selected:", selected_features.shape)

        result = trainer.train(
            epochs=envelop_epochs,
            features=selected_features,
            model_name="dt",
            target_name=model,
        )
        clf = result.estimator

        out_path = Path("data/models")
        os.makedirs(out_path, exist_ok=True)
        file_name = out_path / f"{model}_{MODEL_SUFFIX}"

        print("Mean accuracy after pruning:", result.mean_score)

        clf = cast(DecisionTreeClassifier, result.estimator)

        diag_dir = Path("data/model_diagnostics") / MODEL_SUFFIX

        tree_feature_names = expand_pruned_feature_names(
            epochs=envelop_epochs,
            selected_feature_names=['log_det', 'mav', 'maxav', 'rms', 'ssc', 'std', 'wl'],
            selected_features=selected_features,
            order="feature_major",
        )

        print("Tree feature names:", tree_feature_names)
        print("Number of tree feature names:", len(tree_feature_names))
        print("Number of model input features:", clf.n_features_in_)

        stats = save_tree_diagnostics(
            clf=clf,
            feature_names=tree_feature_names,
            model_name=model,
            mean_score=result.mean_score,
            out_dir=diag_dir,
        )

        print("Tree diagnostics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")

        if TO_SAVE:
            joblib.dump(clf, f"{file_name}.joblib")
            with open(f"{file_name}.pkl", "wb") as f:
                pickle.dump(clf, f)
            print("Saved at:", file_name)
        print()

    print(np.unique(best_features))
    print(['log_det', 'mav', 'maxav', 'rms', 'ssc', 'std', 'wl'])
    print(f"Features min: {np.min(selected_features, axis=0)}")
    print(f"Features max: {np.max(selected_features, axis=0)}")
