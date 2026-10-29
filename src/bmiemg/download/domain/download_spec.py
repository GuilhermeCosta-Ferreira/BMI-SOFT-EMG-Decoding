# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from typing import ClassVar


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class DownloadSpec:
    """Where to download to; `source` selects the strategy."""

    source: ClassVar[str]
    dest: str
