# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from pathlib import Path


# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT: Path = Path(__file__).resolve().parents[1]
DATA: Path = ROOT / "data" / "bids"
DATASET_ROOT: Path = ROOT / "data" / "dataset"
DATASET: Path = DATASET_ROOT / "naive_archive_2026-05-16_epo.fif"

CSV_FILE: Path = ROOT / "emg_record_20260516_200835.csv"


# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    # ------------------------------------------------------------
    # Load MNE epochs
    # ------------------------------------------------------------
    epochs = mne.read_epochs(DATASET, preload=True)

    print(epochs.ch_names)

    channel = "AUX12"

    data = epochs.get_data(picks=[channel])
    # Shape: (n_epochs, 1, n_times)

    signal = data[:, 0, :]
    # Shape: (n_epochs, n_times)

    signal_min = np.min(signal)
    signal_max = np.max(signal)
    signal_mean = np.mean(signal)

    print(f"{channel} min:  {signal_min}")
    print(f"{channel} max:  {signal_max}")
    print(f"{channel} mean: {signal_mean}")

    # ------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------
    df = pd.read_csv(CSV_FILE)

    csv_channel = "ch0"

    csv_signal = df[csv_channel]

    csv_min = csv_signal.min()
    csv_max = csv_signal.max()
    csv_mean = csv_signal.mean()

    print(f"{csv_channel} min:  {csv_min}")
    print(f"{csv_channel} max:  {csv_max}")
    print(f"{csv_channel} mean: {csv_mean}")

    # ------------------------------------------------------------
    # Plot MNE epochs with MNE browser
    # ------------------------------------------------------------
    # ------------------------------------------------------------
    # Plot one MNE epoch channel
    # ------------------------------------------------------------
    epoch_idx = 100

    mne_time_ms = epochs.times * 1000
    mne_signal = signal[epoch_idx]

    plt.figure()
    plt.plot(mne_time_ms, mne_signal)
    plt.xlabel("Time [ms]")
    plt.ylabel(channel)
    plt.title(f"MNE epoch {epoch_idx}: {channel}")
    plt.grid(True)
    plt.show(block=False)

    # ------------------------------------------------------------
    # Plot CSV signal
    # ------------------------------------------------------------
    time_ms = df["timestamp_ms"] - df["timestamp_ms"].iloc[0]

    plt.figure()
    plt.plot(time_ms, df[csv_channel])
    plt.xlabel("Time [ms]")
    plt.ylabel(csv_channel)
    plt.title(f"CSV signal: {csv_channel}")
    plt.grid(True)
    plt.show()
