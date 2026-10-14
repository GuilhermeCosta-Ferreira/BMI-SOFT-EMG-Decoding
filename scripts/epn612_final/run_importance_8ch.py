"""Extract final filtered windows and fit the eight-channel importance tree."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import argparse
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo
import json
import time
from scripts.epn612_final import run_preprocessed_windowing as windows
import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="New output directory (must not exist)"
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    output = (
        args.output
        or windows.BASE
        / "results"
        / datetime.now(ZoneInfo("Europe/Zurich")).strftime(
            "highpass20_notch50_importance_8ch_%Y-%m-%d_%H-%M-%S_%f"
        )
    ).resolve()
    output.mkdir(parents=True, exist_ok=False)
    features_dir = output / "extracted_features"
    features_dir.mkdir()
    analysis = windows.analysis
    labels = analysis.get_training_labels(analysis.TRAINING_DIR, analysis.ARCHIVE_PATH)
    tree = analysis.DecisionTreeFactory(random_state=42).create()
    config = {
        "status": "extracting_features",
        "source": str(analysis.ARCHIVE_PATH),
        "fs_hz": 200,
        "channels": list(windows.EPN612_CHANNELS),
        "window_ms": 200,
        "step_ms": 100,
        "overlap_percent": 50,
        "processing_order": "full raw recording -> 50 Hz notch -> 20 Hz high-pass -> existing windows -> existing annotation selection -> canonical features",
        "filter_parameters": {
            "notch_hz": 50,
            "notch_q": 30,
            "highpass_hz": 20,
            "highpass_order": 4,
            "zero_phase": True,
        },
        "channel_pruning": False,
        "binary_trees_trained": 0,
        "evaluation_run": False,
        "importance_method": "Existing Phase 1 multiclass ranking tree on both original labeled fields; impurity importance summed across eight channels per feature type",
        "labels": labels,
        "feature_types": list(windows.FEATURE_TYPES),
        "tree_parameters": tree.get_params(),
    }

    def checkpoint(status):
        config["status"] = status
        config["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    print("Results:", output, flush=True)
    checkpoint("extracting_features")
    try:
        if args.validate_only:
            windows.validate(output, labels)
            checkpoint("validation_complete")
            return
        storage, matrices, targets, extraction = windows.extract(features_dir, labels)
        config["extraction"] = extraction
        checkpoint("fitting_general_importance_tree")
        x = pd.DataFrame(storage, columns=windows.NAMES, copy=False)
        y = np.concatenate([targets[field] for field in windows.FIELDS])
        print(
            f"Fitting one general importance tree: {len(y):,} windows, 192 inputs",
            flush=True,
        )
        tree.fit(x, y)
        if list(tree.feature_names_in_) != windows.NAMES:
            raise ValueError("Tree feature order mismatch")
        metadata = analysis.get_feature_df(
            SimpleNamespace(ch_names=windows.EPN612_CHANNELS),
            [*windows.TIME_FEATURE_FUNCTIONS, *windows.FREQ_FEATURE_FUNCTIONS],
        )
        if metadata.feature.tolist() != windows.NAMES:
            raise ValueError("Importance metadata order mismatch")
        ranking = windows.custom._feature_type_importance(
            metadata, tree.feature_importances_, 1.0
        )
        metadata.assign(importance=tree.feature_importances_).to_csv(
            output / "raw_feature_channel_importance_8ch.csv", index=False
        )
        ranking.to_csv(output / "feature_importance_8ch.csv", index=False)
        joblib.dump(tree, output / "general_multiclass_importance.joblib")
        windows.feature_plots._plot_feature_types(
            ranking,
            "EPN612 | General feature importance | 8 channels",
            "20 Hz high-pass + 50 Hz notch | 200 ms windows, 50% overlap",
            f"Multiclass ranking tree on {len(y):,} annotated windows; impurity importance summed across all 8 channels.",
            output / "feature_importance_8ch.png",
        )
        config["fitted_tree_depth"] = tree.get_depth()
        config["fitted_tree_nodes"] = tree.tree_.node_count
        checkpoint("complete")
        print(ranking.to_string(index=False), flush=True)
        print(f'COMPLETE ({config["elapsed_seconds"]:.1f}s): {output}', flush=True)
    except Exception as error:
        config["error"] = f"{type(error).__name__}: {error}"
        checkpoint("failed")
        raise


if __name__ == "__main__":
    main()
