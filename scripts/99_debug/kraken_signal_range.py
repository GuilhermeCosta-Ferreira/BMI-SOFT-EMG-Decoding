# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pyxdf

import numpy as np

from pathlib import Path



# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT: Path = Path(__file__).resolve().parents[2]
RAW: Path = ROOT / "data" / "raw"

# Kraken front-end ADC characteristics. Values in the .xdf are raw ADC counts
# (no unit/gain metadata is stored in the stream), so the range must be judged
# against the converter's full scale rather than an absolute voltage.
ADC_BITS: int = 14                       # 14-bit -> codes 0 .. 16383
ADC_MAX: int = (1 << ADC_BITS) - 1       # 16383
ADC_MID: float = ADC_MAX / 2.0           # expected zero-EMG baseline (~8191)

# A sample is counted as "clipping" if it lands within this fraction of a rail.
CLIP_MARGIN: float = 0.01                 # within 1% of 0 or ADC_MAX



# ================================================================
# 2. Section: FUNCTIONS
# ================================================================
def load_emg_stream(path: Path) -> np.ndarray:
    """Return the (n_samples, n_channels) EMG array from a Kraken .xdf file."""
    streams, _ = pyxdf.load_xdf(str(path))
    emg = [s for s in streams if s["info"]["type"][0].upper() == "EMG"]
    if not emg:
        raise ValueError(f"No EMG stream in {path.name}")
    return np.asarray(emg[0]["time_series"], dtype=np.float64)


def report(name: str, data: np.ndarray) -> None:
    """Print per-channel range / headroom / clipping for one recording."""
    lo_rail = CLIP_MARGIN * ADC_MAX
    hi_rail = ADC_MAX - CLIP_MARGIN * ADC_MAX

    print(f"\n### {name}   shape={data.shape}")
    header = f"{'ch':>3} | {'min':>7} | {'max':>7} | {'peak|x-mid|':>11} | {'%full-scale':>11} | {'%clipped':>8}"
    print(header)
    print("-" * len(header))

    for c in range(data.shape[1]):
        col = data[:, c]
        cmin, cmax = col.min(), col.max()
        # Largest swing away from the baseline, expressed against half-scale.
        peak = max(abs(cmax - ADC_MID), abs(cmin - ADC_MID))
        pct_fs = 100.0 * peak / ADC_MID
        clipped = np.mean((col <= lo_rail) | (col >= hi_rail)) * 100.0
        print(f"{c:>3} | {cmin:7.0f} | {cmax:7.0f} | {peak:11.0f} | {pct_fs:10.1f}% | {clipped:7.2f}%")



# ================================================================
# 3. Section: MAIN
# ================================================================
if __name__ == "__main__":
    files = sorted(RAW.rglob("*_emg_kraken.xdf"))
    if not files:
        raise SystemExit(f"No Kraken .xdf files found under {RAW}")

    print(f"ADC model: {ADC_BITS}-bit, codes 0..{ADC_MAX}, baseline ~{ADC_MID:.0f}")
    print(f"Clip threshold: within {CLIP_MARGIN:.0%} of either rail")

    for f in files:
        try:
            data = load_emg_stream(f)
        except ValueError as err:
            print(f"\n### {f.name}: {err}")
            continue
        report(f.relative_to(RAW).as_posix(), data)
