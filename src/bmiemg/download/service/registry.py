# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from ..domain import DownloadStrategy



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Registry:
    def register(self, source: str, strategy: DownloadStrategy) -> None:
        pass

    def get(self, source: str) -> DownloadStrategy:
        pass
