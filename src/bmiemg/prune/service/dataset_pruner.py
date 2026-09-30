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
    source_name: list[str]

    _data_dir: Path = Path("data/")
    _protocol_registry: ProtocolRegistry = field(default_factory=ProtocolRegistry)

    def post_init(self):
        self._source = Source(data_dir=self._data_dir)

    def run(self, protocol_name: str):
        protocol = self._protocol_registry.get(protocol_name)

        return protocol.apply(...)
