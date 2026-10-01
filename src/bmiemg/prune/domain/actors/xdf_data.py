# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from dataclasses import dataclass

from .data_actor import DataActor


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class XdfData(DataActor):
    streams: list[dict]
    file_header: dict
    file_name: str
    file_path: Path
