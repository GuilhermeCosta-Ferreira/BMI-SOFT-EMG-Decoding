# ================================================================
# 0. Section: IMPORTS
# ================================================================
import requests

from dataclasses import dataclass
from requests.auth import HTTPBasicAuth

from ...domain import DownloadStrategy
from .archive_specs import ArchiveSpecs


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ArchiveStrategy(DownloadStrategy):
    def validate(self, spec: ArchiveSpecs) -> None:
        self._auth = HTTPBasicAuth(spec.username, spec.password)

        response = requests.request(
            method="PROPFIND",
            url=spec._dav_root,
            headers=spec._header,
            data=spec._body,
            auth=self._auth,
            allow_redirects=False,
        )

        if response.status_code != 207:
            raise ValueError(f"Unexpected status code: {response.status_code}")



    def fetch(self, spec: ArchiveSpecs) -> None:
        raise NotImplementedError
