# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from dataclasses import dataclass, field

from ..adapters import Source, Loader
from ..domain import ProtocolRegistry



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DatasetPruner:
    source_name: list[str]

    _data_dir: Path = Path("data/")
    _protocol_registry: ProtocolRegistry = field(default_factory=ProtocolRegistry)

    def __post_init__(self):
        self._source = Source(data_dir=self._data_dir)
        self._loader = Loader(source=self._source)

    def run(self, protocol_name: str):
        #protocol = self._protocol_registry.get(protocol_name)
        self._loader.load(self.source_name)
        #return protocol.apply(...)
