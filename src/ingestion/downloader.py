"""
downloader.py
Lists files in a public Google Drive folder first,
then downloads only the first MAX_PDFS PDFs — nothing more.
"""

import os
from pathlib import Path

import gdown
import structlog

log = structlog.get_logger(__name__)

GDRIVE_FOLDER_ID = os.getenv("GDRIVE_FOLDER_ID", "1yxhF1lFF2gKeTNc8Wh0EyBdMT3M4pDYr")
MAX_PDFS = 5
RAW_DIR = Path("data/raw")



def download_pdfs(
    folder_id: str = GDRIVE_FOLDER_ID,
    dest: Path = RAW_DIR,
) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)

    existing = sorted(dest.glob("**/*.pdf"))
    if existing:
        log.info("using_local_pdfs", count=len(existing[:MAX_PDFS]))
        return existing[:MAX_PDFS]

    # Only hit Drive if data/raw/ is completely empty
    log.info("no_local_pdfs_attempting_drive_download")

    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"

    # List files without downloading
    all_files = gdown.download_folder(
        url=folder_url,
        output=str(dest),
        quiet=True,
        use_cookies=False,
        skip_download=True,
        remaining_ok=True,
    )
    if not all_files:
        log.warning("drive_folder_empty_or_inaccessible", folder_id=folder_id)
        return []

    pdf_files = [f for f in all_files if f.path.lower().endswith(".pdf")][:MAX_PDFS]
    log.info("drive_pdfs_selected", count=len(pdf_files))

    downloaded: list[Path] = []
    for f in pdf_files:
        out_path = dest / Path(f.path).name
        result = gdown.download(id=f.id, output=str(out_path), quiet=False, use_cookies=False)
        if result:
            downloaded.append(Path(result))
            log.info("pdf_downloaded", name=out_path.name)
        else:
            log.warning("pdf_download_failed", name=out_path.name)

    return downloaded