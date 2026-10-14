# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass(frozen=True)
class ExportContext:
    protocol_name: str
    timestamp: str
    fmt: str
    zip_groups: bool
