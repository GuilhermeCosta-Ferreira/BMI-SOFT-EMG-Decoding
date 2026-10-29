# Importing adapters registers every strategy before Downloader looks one up.
from .adapters import ArchiveSpecs, EPN612Specs
from .service import Downloader

__all__ = [
    "ArchiveSpecs",
    "Downloader",
    "EPN612Specs",
]
