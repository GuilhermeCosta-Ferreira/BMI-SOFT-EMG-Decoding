# ================================================================
# 0. Section: IMPORTS
# ================================================================
import mne

import numpy as np

from pathlib import Path
from mne.io.constants import FIFF



# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT: Path = Path(__file__).resolve().parents[2]
DATA: Path = ROOT / "data" / "bids"
DATASET_ROOT: Path = ROOT / "data" / "dataset"
DATASET: Path = DATASET_ROOT / "naive_archive_2026-05-12_epo.fif"

# Full-scale input range of the in-house ADC/front-end AFTER the gain stage,
# expressed as a single-sided amplitude in volts (i.e. +/- ADC_FS_V).
# The gain must map the largest EMG excursion to (just under) this value.
ADC_FS_V: float = 2.5

# Fraction of the ADC range we allow the biggest EMG peak to occupy, leaving
# headroom for artifacts / inter-subject variability (0.8 -> keep 20% margin).
HEADROOM: float = 0.8



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
UNIT_NAMES = {FIFF.FIFF_UNIT_V: "V", FIFF.FIFF_UNIT_T: "T", FIFF.FIFF_UNIT_T_M: "T/m"}


def suggest_gain(peak_v: float) -> float:
    """Gain needed so a signal of amplitude `peak_v` fills HEADROOM of the ADC range."""
    return (HEADROOM * ADC_FS_V) / peak_v



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == '__main__':
    epochs = mne.read_epochs(DATASET, preload=True)

    unit = UNIT_NAMES.get(epochs.info["chs"][0]["unit"], "unknown")
    print(f"Stored channel unit: {unit}   |   channel types: {set(epochs.get_channel_types())}")

    # (epochs, channels, times) in millivolts
    data_mv = epochs.get_data(units="mV")
    ch_names = epochs.ch_names

    # Per-channel stats across all epochs/samples. Global min/max is dominated by
    # single outliers, so we also report the 99.9th percentile of |signal| as a
    # robust "practical peak" for gain sizing.
    print(f"\n{'channel':>10} | {'min(mV)':>9} | {'max(mV)':>9} | {'|peak|(mV)':>10} | {'p99.9|x|(mV)':>12} | {'gain@|peak|':>11}")
    print("-" * 82)

    for ci, name in enumerate(ch_names):
        ch = data_mv[:, ci, :]
        cmin = float(np.nanmin(ch))
        cmax = float(np.nanmax(ch))
        peak = max(abs(cmin), abs(cmax))
        p999 = float(np.nanpercentile(np.abs(ch), 99.9))
        gain = suggest_gain(peak / 1000.0)  # convert mV -> V
        print(f"{name:>10} | {cmin:9.3f} | {cmax:9.3f} | {peak:10.3f} | {p999:12.3f} | {gain:11.1f}")

    # Global figures
    gmin = float(np.nanmin(data_mv))
    gmax = float(np.nanmax(data_mv))
    gpeak = max(abs(gmin), abs(gmax))
    gp999 = float(np.nanpercentile(np.abs(data_mv), 99.9))

    print("-" * 82)
    print(f"\nGlobal raw range : {gmin:.3f} mV  ..  {gmax:.3f} mV")
    print(f"Global |peak|    : {gpeak:.3f} mV   (worst-case single sample)")
    print(f"Global p99.9 |x| : {gp999:.3f} mV   (robust practical peak)")

    print(f"\nADC assumption   : +/- {ADC_FS_V} V full scale, keeping {HEADROOM:.0%} headroom")
    print(f"Gain for worst-case peak : {suggest_gain(gpeak / 1000.0):.1f} x")
    print(f"Gain for p99.9 peak      : {suggest_gain(gp999 / 1000.0):.1f} x")
