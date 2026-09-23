# ================================================================
# 0. Section: IMPORTS
# ================================================================
from typing import ClassVar
from dataclasses import dataclass

from ...domain import DownloadSpec


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class EPN612Specs(DownloadSpec):
    source: ClassVar[str] = "emg-epn-612"

    record_id: int = 4421500
    filename: str = "EMG-EPN612 Dataset.zip"
