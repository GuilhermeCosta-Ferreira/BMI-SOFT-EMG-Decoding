# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass

from .metadata_actor import MetadataActor


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass(kw_only=True)
class XdfData(MetadataActor):
    streams: list[dict]
