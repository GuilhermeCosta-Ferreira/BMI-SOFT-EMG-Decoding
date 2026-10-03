# ================================================================
# 0. Section: IMPORTS
# ================================================================
from datetime import datetime

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
    directory_structure: str = "bids_local" # this assumes the current archive structure
    protocol_version: int = 0 # 0 is undefined

    @property
    def session_folder(self) -> Path | None:
        if self.directory_structure == "bids_local":
            return self.file_path.parent.parent
        return None

    @property
    def session_number(self) -> int | None:
        if self.directory_structure == "bids_local":
            return int(self.session_folder.name[3:])
        return None

    @property
    def subject_folder(self) -> Path | None:
        if self.directory_structure == "bids_local":
            return self.session_folder.parent
        return None

    @property
    def subject_number(self) -> int | None:
        if self.directory_structure == "bids_local":
            return int(self.subject_folder.name[3:])
        return None

    @property
    def date_folder(self) -> Path | None:
        if self.directory_structure == "bids_local":
            return self.subject_folder.parent
        return None

    @property
    def date(self) -> datetime | None:
        if self.date_folder:
            return datetime.fromisoformat(self.date_folder.name)
        return None
