# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field

from ..domain import DownloadSpec, Registry


# ================================================================
# 1. Section: Class definition
# ================================================================
@dataclass
class Downloader:
    """Runs the strategy registered for a spec's source."""

    _registry: Registry = field(default_factory=Registry)

    def run(self, spec: DownloadSpec) -> None:
        """Download `spec` with its registered strategy."""
        # 1. Select strategy based on source
        source = spec.source
        strategy = self._registry.get(source)

        # 2. Fetch data using selected strategy
        strategy.fetch(spec)
