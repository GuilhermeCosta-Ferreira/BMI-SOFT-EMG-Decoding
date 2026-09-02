# ================================================================
# 0. Section: IMPORTS
# ================================================================
from dataclasses import dataclass, field

from ...domain import DownloadSpec


# ================================================================
# 1. Section: Functions
# ================================================================
@dataclass
class ArchiveSpecs(DownloadSpec):
    username: str
    password: str
    dav_user: str

    _base_url: str = "https://make-archives.epfl.ch"
    _header: dict = field(default_factory=lambda: {
        "Depth": "1",
        "Content-Type": "application/xml",
    })
    _body: str = """<?xml version="1.0"?>
    <d:propfind xmlns:d="DAV:">
      <d:prop>
        <d:displayname />
        <d:resourcetype />
        <d:getcontentlength />
        <d:getlastmodified />
      </d:prop>
    </d:propfind>
    """

    def __post_init__(self):
        self._dav_root = f"{self._base_url}/remote.php/dav/files/{self.dav_user}"
