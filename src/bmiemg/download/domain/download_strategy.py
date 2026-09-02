# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from abc import ABC

from .download_spec import DownloadSpec



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DownloadStrategy (ABC):
    def validate(self, spec: DownloadSpec) -> None:
        raise NotImplementedError

    def fetch(self, spec: DownloadSpec) -> None:
        raise NotImplementedError
