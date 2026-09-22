# ================================================================
# 0. Section: IMPORTS
# ================================================================
import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from .epn612_specs import EPN612Specs
from ...domain import DownloadStrategy, Registry


CHUNK_SIZE = 1024 * 1024
REQUEST_TIMEOUT = (30, 300)

# ================================================================
# 1. Section: Functions
# ================================================================
@Registry.register("emg-epn-612")
@dataclass
class EPN612Strategy(DownloadStrategy[EPN612Specs]):
    def validate(self, spec: EPN612Specs) -> None:
        _resolve_file(spec)

    def fetch(self, spec: EPN612Specs) -> None:
        remote_file = _resolve_file(spec)
        download_url = _download_url(remote_file)
        expected_size = int(remote_file["size"])
        expected_checksum = _md5_checksum(remote_file["checksum"])

        destination = Path(spec.dest)
        destination.mkdir(parents=True, exist_ok=True)

        archive_path = destination / spec.filename
        partial_path = archive_path.with_suffix(archive_path.suffix + ".part")

        # A valid archive means the dataset was already downloaded and is up-to-date => skip download
        if not _is_valid_file(archive_path, expected_size, expected_checksum):
            actual_checksum = _stream_download(
                download_url,
                partial_path,
                expected_size,
                spec.filename,
            )
            actual_size = partial_path.stat().st_size

            if actual_size != expected_size:
                raise ValueError(
                    f"Size mismatch for {spec.filename!r}: expected "
                    f"{expected_size} bytes, received {actual_size} bytes"
                )

            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"Checksum mismatch for {spec.filename!r}: expected "
                    f"{expected_checksum}, received {actual_checksum}"
                )

            # Only expose the final filename after all integrity checks pass.
            # The .part file remains available for diagnosis after a failure.
            partial_path.replace(archive_path)

        print(f"Extracting {spec.filename} . This can take several minutes...")
        extraction_path = destination / archive_path.stem
        _extract_zip_safely(archive_path, extraction_path)
        print(f"Done.")


# ──────────────────────────────────────────────────────
# 1.1 Subsection: Helper Functions
# ──────────────────────────────────────────────────────
def _resolve_file(spec: EPN612Specs) -> dict:
    metadata = _get_metadata(spec.record_id)
    return _find_file(metadata, spec.filename)


def _get_metadata(record_id: int) -> dict:
    url = f"https://zenodo.org/api/records/{record_id}"
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _find_file(metadata: dict, filename: str) -> dict:
    for remote_file in metadata.get("files", []):
        if remote_file.get("key") == filename:
            return remote_file

    available_files = [
        remote_file.get("key") for remote_file in metadata.get("files", [])
    ]
    raise ValueError(
        f"File {filename!r} was not found in the Zenodo record. "
        f"Available files: {available_files}"
    )


def _download_url(remote_file: dict) -> str:
    links = remote_file.get("links", {})
    url = links.get("content") or links.get("download") or links.get("self")
    if not url:
        raise ValueError(f"No download URL was provided for {remote_file.get('key')!r}")
    return url


def _md5_checksum(checksum: str) -> str:
    algorithm, separator, value = checksum.partition(":")
    if separator != ":" or algorithm.lower() != "md5" or not value:
        raise ValueError(f"Expected a Zenodo MD5 checksum, received {checksum!r}")
    return value.lower()


def _stream_download(
    url: str,
    destination: Path,
    expected_size: int,
    description: str,
) -> str:
    digest = hashlib.md5()

    with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()

        with (
            destination.open("wb") as output,
            tqdm(
                total=expected_size,
                desc=description,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue

                output.write(chunk)
                digest.update(chunk)
                progress.update(len(chunk))

    return digest.hexdigest()


def _is_valid_file(path: Path, expected_size: int, expected_checksum: str) -> bool:
    if not path.is_file() or path.stat().st_size != expected_size:
        return False
    return _file_md5(path) == expected_checksum


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_zip_safely(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    extraction_root = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        # Commented for performance purpose: probably redundant integrity test, and doubles unzipping time.
        #corrupt_member = archive.testzip() # check CRC checksum (zip)
        #if corrupt_member is not None:
        #    raise ValueError(f"Corrupted ZIP member: {corrupt_member}")

        for member in archive.infolist():
            member_path = (extraction_root / member.filename).resolve()
            if not member_path.is_relative_to(extraction_root): # check that every member's path is under the extraction root
                raise ValueError(f"Unsafe ZIP member path: {member.filename!r}")

        archive.extractall(extraction_root)
