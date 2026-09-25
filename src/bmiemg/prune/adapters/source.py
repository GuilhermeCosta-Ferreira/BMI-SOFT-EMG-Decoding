# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass
from pathlib import Path



# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class Source:
    data_dir: Path
    is_raw_data: bool = False

    @property
    def dataset_dir(self) -> Path:
        return self.data_dir / "dataset"

    def raw_data_dir(self, dataset_name: str) -> Path:
        if self.is_raw_data:
            return self.data_dir / dataset_name

        return self.dataset_dir / dataset_name

    def out_dir(self, dataset_name: str) -> Path:
        return self.dataset_dir / dataset_name
