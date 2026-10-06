# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from datetime import date
from dataclasses import dataclass

from .data_actor import DataActor


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
