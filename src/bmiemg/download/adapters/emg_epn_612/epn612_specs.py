# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar

from ...domain import DownloadSpec


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class EPN612Specs(DownloadSpec):
    """EMG-EPN-612 archive on Zenodo."""

    source: ClassVar[str] = "emg-epn-612"

    record_id: int = 4421500
    filename: str = "EMG-EPN612 Dataset.zip"
