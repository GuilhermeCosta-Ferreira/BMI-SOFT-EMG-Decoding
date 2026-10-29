# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path

from bmiemg.download import Downloader, EPN612Specs

# ================================================================
# 1. Section: INPUTS
# ================================================================
ROOT = Path(__file__).resolve().parents[2]


# ================================================================
# 2. Section: MAIN
# ================================================================
if __name__ == "__main__":
    specs = EPN612Specs(
        dest=str(ROOT / "data" / "raw" / "emg_epn_612"),
    )

    downloader = Downloader()
    downloader.run(specs)
