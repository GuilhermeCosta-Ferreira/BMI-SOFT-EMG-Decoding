# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from .data_actor import DataActor

Signal = Literal["emg", "eeg"]


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass(kw_only=True)
class MetadataActor(DataActor):
    file_header: dict
    file_name: str
    file_path: Path
    directory_structure: str = "bids_local"  # assumes the current archive structure
    protocol_version: int = 0  # 0 is undefined
    signal_type: Signal | None = None
    # Set by loaders whose sources carry no bids folder layout to derive it from.
    subject_id: int | None = None

    # ================================================================
    # 2. Section: PROPERTIES
    # ================================================================
    @property
    def session_folder(self) -> Path | None:
        if self.directory_structure == "bids_local":
            return self.file_path.parent.parent
        return None

    @property
    def session_number(self) -> int | None:
        folder = self.session_folder
        if folder is not None:
            return int(folder.name[-2:])
        return None

    @property
    def subject_folder(self) -> Path | None:
        folder = self.session_folder
        if folder is not None:
            return folder.parent
        return None

    @property
    def subject_number(self) -> int | None:
        if self.subject_id is not None:
            return self.subject_id
        folder = self.subject_folder
        if folder is not None:
            return int(folder.name[-2:])
        return None

    @property
    def date_folder(self) -> Path | None:
        folder = self.subject_folder
        if folder is not None:
            return folder.parent
        return None

    @property
    def date(self) -> date | None:
        folder = self.date_folder
        if folder is not None:
            return date.fromisoformat(folder.name)
        return None
