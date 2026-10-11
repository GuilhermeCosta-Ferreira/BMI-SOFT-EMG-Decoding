# ================================================================
# 0. Section: IMPORTS
# ================================================================
from bmiemg.utils import Registry

from .writer import Writer



# ================================================================
# 1. Section: Class definition
# ================================================================
class WriterRegistry(Registry[Writer]): ...
