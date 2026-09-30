# ================================================================
# 0. Section: IMPORTS
# ================================================================
import pyxdf

from pathlib import Path

from .file_loader import FileLoader
from ...domain import DataActor, XdfData
from .file_loader_registry import FileLoaderRegistry



# ================================================================
# 1. Section: Class definition
# ================================================================
@FileLoaderRegistry.register(".xdf")
class XdfLoader(FileLoader):
    def load_file(self, path: Path) -> DataActor:
        streams, file_header = pyxdf.load_xdf(path)

        return XdfData(streams, file_header, path.stem, path)
