# ================================================================
# 0. Section: IMPORTS
# ================================================================
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .download_spec import DownloadSpec


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class DownloadStrategy[SpecT: DownloadSpec](ABC):
    """Fetches one source described by its spec."""

    @abstractmethod
    def validate(self, spec: SpecT) -> None:
        """Raise if the source cannot be reached with this spec."""
        raise NotImplementedError

    @abstractmethod
    def fetch(self, spec: SpecT) -> None:
        """Download the source to `spec.dest`."""
        raise NotImplementedError
