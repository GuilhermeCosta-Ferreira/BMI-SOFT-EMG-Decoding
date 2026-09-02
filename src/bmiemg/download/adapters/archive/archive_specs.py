# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ...domain import DownloadSpec


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ArchiveSpecs(DownloadSpec):
    username: str
    password: str
    key: str
