# ================================================================
# 0. Section: IMPORTS
# ================================================================
from tqdm import tqdm
from typing import Any
from pathlib import Path
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..adapters import Source, Loader
from ..domain import ProtocolRegistry, PruneStep, PruneSpecs, DataActor


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class DatasetPruner:
    source_name: list[str]

    custom_steps: list[PruneStep] = field(default_factory=list)
    custom_specs: list[PruneSpecs] = field(default_factory=list)

    _data_dir: Path = Path("data/")
    _protocol_registry: ProtocolRegistry = field(default_factory=ProtocolRegistry)

    def __post_init__(self):
        self._source = Source(data_dir=self._data_dir)
        self._loader = Loader(source=self._source)

    # ================================================================
    # 2. Section: MAIN FUNCTIONS
    # ================================================================
    def run(self, protocol_name: str | None) -> list[DataActor]:
        # 1. Load all the actors into the scene
        actors = self._loader.load(self.source_name)

        # 2. Load all the steps into the scene
        steps, specs = self._load_steps(protocol_name)

        # 3. Apply the steps to the actors
        pruned_actors = self.apply(actors, steps, specs)

        return pruned_actors

    def apply(
        self,
        actors: list[DataActor],
        steps: Sequence[PruneStep[Any, Any, Any]],
        specs: Sequence[PruneSpecs],
    ) -> list[DataActor]:
        for step, spec in tqdm(
            zip(steps, specs), total=len(steps), desc="Applying steps"
        ):
            actors = step.apply(actors)

        return actors

    # ──────────────────────────────────────────────────────
    # 1.1 Subsection: Helper Functions
    # ──────────────────────────────────────────────────────
    def _load_steps(
        self, protocol_name: str | None
    ) -> tuple[Sequence[PruneStep[Any, Any, Any]], Sequence[PruneSpecs]]:
        if self.custom_steps and self.custom_specs:
            steps = self.custom_steps
            specs = self.custom_specs
        elif protocol_name is not None:
            steps = self._protocol_registry.get(protocol_name).steps
            specs = self._protocol_registry.get(protocol_name).specs
        else:
            raise ValueError("No protocol name provided and no custom steps defined")

        return steps, specs
