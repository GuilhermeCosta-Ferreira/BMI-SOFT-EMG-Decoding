# Download

Layers: `service` (`Downloader`) → `domain` (specs, strategies, registry) → `adapters` (one per source). To add a source, subclass `DownloadStrategy` + `DownloadSpec` under `adapters/` and decorate the strategy with `@Registry.register("<source-name>")`.

## Archive source

Pulls files from EPFL's WebDAV archive (`make-archives.epfl.ch`).

1. Install dependencies: `poetry install`.
2. Copy `.env.example` to `.env` and fill in your credentials (never commit them):

   ```env
   MAKER_USERNAME=<your username>
   MAKER_APP_PASSWORD=<your app password>
   MAKER_DAV_USER=<your dav user>
   ```

3. Set the remote `url` and local `dest` in `scripts/download/download_archive.py`, then run it (it loads `.env` automatically):

   ```bash
   poetry run python scripts/download/download_archive.py
   ```

### Getting the credentials

- **`MAKER_USERNAME`** — Is your EPFL username
- **`MAKER_APP_PASSWORD`** — Go to Profile Settings (top right) on the Archive MAKE Drive > Security > Create New App Password
- **`MAKER_DAV_USER`** — Go to Files Settings (bottom left) on the Archive MAKE drive > WebDAV > Copy the URL and past the code in the URL that comes after https://make-archives.epfl.ch/remote.php/dav/files/
