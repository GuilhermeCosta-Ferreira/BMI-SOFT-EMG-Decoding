# ================================================================
# 0. Section: IMPORTS
# ================================================================
from bmiemg.utils import Registry

from .protocols import Protocol


# ================================================================
# 1. Section: Class definition
# ================================================================
class ProtocolRegistry(Registry[Protocol]):...
