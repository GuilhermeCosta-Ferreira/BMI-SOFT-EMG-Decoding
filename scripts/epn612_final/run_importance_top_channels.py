"""Train Top-4/Top-6 gesture trees using cached final high-pass features."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import argparse
from types import SimpleNamespace
import json
import os
import time
from scripts.epn612_final import run_preprocessed_windowing as windows
import joblib
import numpy as np
import pandas as pd


def channel_rank(raw):
    ranking = raw.groupby("channel", as_index=False, sort=False).importance.sum()
    ranking = ranking.sort_values(
        "importance", ascending=False, kind="stable"
    ).reset_index(drop=True)
    if not np.isclose(ranking.importance.sum(), 1):
        raise ValueError("Channel importance must sum to one")
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    return ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Output directory from run_importance_8ch.py",
    )
    parser.add_argument("--n-jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.n_jobs < 1:
        parser.error("--n-jobs must be positive")
    source = args.source.resolve()
    output = source / "channel_top4_top6_importance"
    output.mkdir(exist_ok=True)
    analysis = windows.analysis
    previous = json.loads((source / "config.json").read_text())
    if previous["filter_parameters"] != {
        "notch_hz": 50,
        "notch_q": 30,
        "highpass_hz": 20,
        "highpass_order": 4,
        "zero_phase": True,
    }:
        raise ValueError("Expected 20 Hz high-pass + 50 Hz notch features")
    if previous["status"] != "complete":
        raise ValueError("Cached high-pass extraction must be complete")
    started = time.monotonic()
    params = analysis.DecisionTreeFactory(random_state=42).create().get_params()
    if params != previous["tree_parameters"]:
        raise ValueError("Tree settings differ from the preceding high-pass run")
    config = {
        "status": "channel_importance",
        "cached_feature_source": str(source),
        "filter_parameters": previous["filter_parameters"],
        "window_ms": 200,
        "step_ms": 100,
        "fs_hz": 200,
        "tree_parameters": params,
        "n_jobs": args.n_jobs,
        "extraction_repeated": False,
        "evaluation_run": False,
        "selection_method": "Fixed Top-4 / Top-6 from the cached general eight-channel ranking tree",
        "importance_method": "Existing independent one-versus-all binary trees trained on trainingSamples windows; sum impurity importance across selected channels per feature type; global values average six trees equally",
        "configurations": {},
    }

    def checkpoint(status):
        config["status"] = status
        config["elapsed_seconds"] = round(time.monotonic() - started, 2)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    def fit_one(label, x, y, model_path):
        name = model_path.parent.parent.name
        print(f"{name}: fitting {label} versus all", flush=True)
        before = time.monotonic()
        if model_path.exists():
            tree = joblib.load(model_path)
        else:
            tree = analysis.DecisionTreeFactory(random_state=42).create()
            tree.fit(x, (y == label).astype(np.int8))
            joblib.dump(tree, model_path)
        if (
            tree.get_params() != params
            or list(tree.feature_names_in_) != list(x.columns)
            or not np.array_equal(tree.classes_, [0, 1])
        ):
            raise ValueError(f"Invalid fitted {name}/{label} model")
        print(f"{name}: fitted {label} in {time.monotonic() - before:.1f}s", flush=True)
        return label, tree

    print("Results:", output, flush=True)
    checkpoint("channel_importance")
    try:
        raw = pd.read_csv(source / "raw_feature_channel_importance_8ch.csv")
        ranked = channel_rank(raw)
        ranked.to_csv(output / "channel_importance_8ch.csv", index=False)
        analysis._save_importance_plot(
            ranked,
            "channel",
            output / "channel_importance_8ch.png",
            "EPN612 | 8 channels | 20 Hz high-pass + 50 Hz notch | 200 ms",
        )
        channels = tuple(ranked.channel)
        config["ranked_channels"] = list(channels)
        print(ranked.to_string(index=False), flush=True)

        labels = previous.get("labels")
        if labels is None:
            labels = analysis.get_training_labels(
                analysis.TRAINING_DIR, analysis.ARCHIVE_PATH
            )
        ntrain = previous["extraction"]["windows"]["trainingSamples"]
        full = np.load(source / "extracted_features/features.npy", mmap_mode="r")
        codes = np.load(source / "extracted_features/label_codes.npy", mmap_mode="r")
        all_names = analysis.get_feature_names(windows.EPN612_CHANNELS)
        if full.shape != (
            sum(previous["extraction"]["windows"].values()),
            len(all_names),
        ) or codes.shape != (len(full),):
            raise ValueError("Cached matrices do not match extraction provenance")
        y = np.asarray(labels)[codes[:ntrain]]
        if set(y) != set(labels):
            raise ValueError("Incomplete training labels")

        for size in (4, 6):
            key = f"top{size}"
            selected = channels[:size]
            names = analysis.get_feature_names(selected)
            folder = output / key
            (folder / "models").mkdir(parents=True, exist_ok=True)
            per_gesture = folder / "feature_importance" / "by_feature_type"
            per_gesture.mkdir(parents=True, exist_ok=True)
            metadata = analysis.get_feature_df(
                SimpleNamespace(ch_names=selected),
                [*windows.TIME_FEATURE_FUNCTIONS, *windows.FREQ_FEATURE_FUNCTIONS],
            )
            if metadata.feature.tolist() != names:
                raise ValueError("Feature metadata order mismatch")
            # Trees already convert inputs to float32. Cache that same representation
            # once and share it across fitting threads, avoiding repeated copies.
            indices = [all_names.index(name) for name in names]
            x = pd.DataFrame(
                np.asarray(full[:ntrain, indices], dtype=np.float32),
                columns=names,
                copy=False,
            )
            if x.shape != (ntrain, size * 24) or not np.isfinite(x.to_numpy()).all():
                raise ValueError("Invalid selected cached features")
            config["configurations"][key] = {
                "channels": list(selected),
                "train_windows": ntrain,
                "model_inputs": len(names),
                "completed_labels": [],
            }
            checkpoint(f"training_{key}")
            print(
                f'{key}: {selected}, {ntrain:,} windows, {len(names)} inputs, {config["n_jobs"]} concurrent fits',
                flush=True,
            )
            models = {}
            results = joblib.Parallel(
                n_jobs=config["n_jobs"],
                prefer="threads",
                return_as="generator_unordered",
            )(
                joblib.delayed(fit_one)(
                    label, x, y, folder / "models" / f"{label}_vs_all.joblib"
                )
                for label in labels
            )
            for label, model in results:
                models[label] = model
                ranking = windows.custom._feature_type_importance(
                    metadata, model.feature_importances_, 1.0
                )
                ranking.to_csv(
                    per_gesture / f"{label}_vs_all_feature_types.csv", index=False
                )
                windows.feature_plots._plot_feature_types(
                    ranking,
                    f"{size}CH | {label} versus all",
                    f'Channels: {", ".join(selected)} | 20 Hz high-pass + 50 Hz notch | 200 ms windows',
                    "Existing binary-tree impurity importance summed across selected channels; all 24 feature types retained.",
                    per_gesture / f"{label}_vs_all_feature_types.png",
                    svg=True,
                )
                config["configurations"][key]["completed_labels"].append(label)
                checkpoint(f"training_{key}")
            ranking = analysis._save_reduced_importance(
                selected, labels, [models[label] for label in labels], folder, key
            )
            windows.feature_plots._plot_feature_types(
                ranking,
                f"{size}CH | Global feature importance",
                f'Channels: {", ".join(selected)} | 20 Hz high-pass + 50 Hz notch | 200 ms windows',
                "Feature importance summed across selected channels, then averaged equally over six gesture-versus-all trees.",
                folder / f"feature_importance_{key}_by_type.png",
                svg=True,
            )
            subset_channels = channel_rank(
                pd.read_csv(folder / f"mean_raw_feature_channel_importance_{key}.csv")
            )
            subset_channels.to_csv(
                folder / f"channel_importance_{key}.csv", index=False
            )
            analysis._save_importance_plot(
                subset_channels,
                "channel",
                folder / f"channel_importance_{key}.png",
                f"EPN612 | {size} channels | Mean binary-tree channel importance",
            )
            config["configurations"][key]["status"] = "complete"
            checkpoint(f"{key}_complete")
            print(
                f"COMPLETE: {key} global and all six per-gesture feature importance plots",
                flush=True,
            )
            del x, models, results
        checkpoint("complete")
        print(f'COMPLETE ({config["elapsed_seconds"]:.1f}s): {output}', flush=True)
    except Exception as error:
        config["error"] = f"{type(error).__name__}: {error}"
        checkpoint("failed")
        raise


if __name__ == "__main__":
    main()
