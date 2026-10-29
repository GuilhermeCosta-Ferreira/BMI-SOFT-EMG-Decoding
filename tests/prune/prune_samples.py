# The only implementation-specific part of the prune contracts: one YAML-shaped config per registered key, as it would appear in configs/prune/*.yaml.
# Values use the types yaml.safe_load produces (e.g. dates, lists for tuples).
from datetime import date

STEP_SAMPLE_CONFIGS: dict[str, dict] = {
    "remove_session": {"files_to_remove": ["sub-01_ses-01_task-x_run-01_raw"]},
    "signal_split": {"signal": "emg"},
    "protocol_definer": {"cuttoff_dates": [date(2025, 11, 21)]},
    "signal_unwarp": {},
    "analogue_filter": {"btype": "bandpass", "cutoff": [20.0, 500.0], "order": 1},
    "marker_unification": {
        "source_schemas": {
            1: {
                "phase_pos": 0,
                "movement_slice": [3, 5],
                "phase_map": {1: "cue"},
                "movement_groups": {"open hand": [1]},
                "special_triggers": {9999: "test marker"},
            }
        },
        "target": {
            "phases": {"undefined": 0, "cue": 1, "test marker": 2},
            "movements": {"undefined": 0, "open hand": 1},
        },
    },
    "build_epochs": {"tmin": -0.5, "tmax": 5.0},
    "ttv_split": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
}

WRITER_SAMPLE_CONFIGS: dict[str, dict] = {
    "dataset": {"fmt": ".fif", "zip_groups": True},
}
