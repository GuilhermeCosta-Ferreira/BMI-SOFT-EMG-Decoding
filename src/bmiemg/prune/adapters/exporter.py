# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from datetime import UTC, datetime

from .source import Source
from ..domain import DataActor
from .exporters import ExportContext, WriterRegistry


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Exporter:
    source: Source

    def export(
        self,
        actors: list[DataActor],
        protocol_name: str,
        fmt: str,
        zip_groups: bool,
    ) -> None:
        out_dir = self.source.out_dir(protocol_name)
        # One timestamp per run so a dataset's files share a suffix.
        ctx = ExportContext(
            protocol_name=protocol_name,
            timestamp=datetime.now(tz=UTC).strftime("%Y-%m-%d_%H-%M-%S"),
            fmt=fmt,
            zip_groups=zip_groups,
        )

        for actor in actors:
            writer = WriterRegistry.get(actor.name)
            writer.write(actor, out_dir, ctx)
