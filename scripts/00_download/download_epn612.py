from pathlib import Path

from bmiemg.download import Downloader, EPN612Specs


ROOT = Path(__file__).resolve().parents[2]

spec = EPN612Specs(
    dest=str(ROOT / "data" / "raw" / "emg_epn_612"),
)

Downloader().run(spec)