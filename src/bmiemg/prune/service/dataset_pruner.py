# ================================================================
# 0. Section: IMPORTS
# ================================================================
from pathlib import Path
from dataclasses import dataclass, field

from ..adapters import Source
from ..domain import ProtocolRegistry



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DatasetPruner:
    dataset_name: str
    is_raw_data: bool

    _source: Source
    _data_dir: Path = Path("data/")
    _protocol_registry: ProtocolRegistry = field(default_factory=ProtocolRegistry)

    def post_init(self):
        self._source = Source(data_dir=self._data_dir, is_raw_data=self.is_raw_data)

    def run(self, protocol_name: str):
        protocol = self._protocol_registry.get(protocol_name)

        return protocol.apply(...)
