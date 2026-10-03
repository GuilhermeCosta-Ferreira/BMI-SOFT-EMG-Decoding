# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .xdf_actor import XdfActor


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass(kw_only=True)
class XdfData(XdfActor):
    streams: list[dict]
