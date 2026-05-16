# ================================================================
# 0. Section: IMPORTS
# ================================================================
import joblib
import pickle

import numpy as np

from pathlib import Path
from dataclasses import dataclass

from .ClassifierEstimator import ClassifierEstimator


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class TrainingResults:
    model_name: str
    target_name: str
    estimator: ClassifierEstimator
    scores: np.ndarray
    mean_score: float
    std_score: float
    channel_names: list[str]
    feature_names: list[str]

    def predict(self, signal: np.ndarray) -> np.ndarray:
        return self.estimator.predict(signal)

    def save(
        self, folder_path: Path, file_type: str = ".joblib", model_name: str = ""
    ) -> Path:
        # 1. Builds the path
        model_name = model_name if model_name else self.model_name
        out_path = folder_path / model_name
        file_path = Path(f"{out_path}{file_type}")

        # 2. Saves the joblib if required
        if file_type == ".joblib":
            joblib.dump(self.estimator, file_path)

        # 3. Saves the pickle if required
        elif file_type == ".pkl":
            with open(file_path, "wb") as f:
                pickle.dump(self.estimator, f)

        # 4. Deals with edge cases
        else:
            raise AttributeError(
                f"No model_name {model_name}. Please select either "
                "'.loblib' or '.pkl'"
            )

        return file_path
