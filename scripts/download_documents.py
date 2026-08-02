"""Script para baixar (via Google Drive API) e filtrar a base documental."""
import io
import os
import time
import shutil
from pathlib import Path

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

load_dotenv()

DRIVE_API_KEY = os.getenv("DRIVE_API_KEY", "")
ROOT_FOLDER_ID = "1mdJJ-a1nXe4wSEMUUI74b5af07x_W-ch"
RAW_DIR = Path("data/CNMP_raw")
DOCS_DIR = Path("data/CNMP_docs")
VALID_EXT = {".pdf", ".txt", ".html", ".htm"}
PAUSA_ENTRE_DOWNLOADS = 0.5

GOOGLE_EXPORT_MIMES = {
    "application/vnd.google-apps.document": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx"),
    "application/vnd.google-apps.spreadsheet": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ".xlsx"),
    "application/vnd.google-apps.presentation": ("application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx"),
}


def list_folder(service, folder_id):
    files, page_token = [], None
    while True:
        response = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields="nextPageToken, files(id, name, mimeType)",
            pageToken=page_token,
            pageSize=1000,
        ).execute()
        files.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return files


def download_file(service, file_id, file_name, mime_type, dest_path, failed):
    if mime_type in GOOGLE_EXPORT_MIMES:
        _, ext = GOOGLE_EXPORT_MIMES[mime_type]
        if not str(dest_path).endswith(ext):
            dest_path = dest_path.with_suffix(ext)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        return

    try:
        if mime_type in GOOGLE_EXPORT_MIMES:
            export_mime, _ = GOOGLE_EXPORT_MIMES[mime_type]
            request = service.files().export_media(fileId=file_id, mimeType=export_mime)
        else:
            request = service.files().get_media(fileId=file_id)

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        dest_path.write_bytes(fh.getvalue())
        print(f"OK: {dest_path}")
    except Exception as exc:
        print(f"FALHOU: {file_name} ({file_id}) -> {exc}")
        failed.append((file_name, file_id, str(exc)))
    finally:
        time.sleep(PAUSA_ENTRE_DOWNLOADS)


def process_folder(service, folder_id, local_path, failed):
    local_path.mkdir(parents=True, exist_ok=True)
    for item in list_folder(service, folder_id):
        name, mime, fid = item["name"], item["mimeType"], item["id"]
        if mime == "application/vnd.google-apps.folder":
            process_folder(service, fid, local_path / name, failed)
        else:
            download_file(service, fid, name, mime, local_path / name, failed)


def download():
    if not DRIVE_API_KEY:
        raise EnvironmentError("DRIVE_API_KEY nao encontrada. Configure o arquivo .env (veja .env.example).")
    service = build("drive", "v3", developerKey=DRIVE_API_KEY)
    failed = []
    process_folder(service, ROOT_FOLDER_ID, RAW_DIR, failed)
    if failed:
        print(f"\n{len(failed)} arquivo(s) falharam:")
        for name, fid, err in failed:
            print(f" - {name} ({fid}): {err}")


def filter_documents():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in sorted(RAW_DIR.rglob("*")):
        if f.is_file() and f.suffix.lower() in VALID_EXT:
            dest = DOCS_DIR / f.name
            if dest.exists():
                dest = DOCS_DIR / f"{f.stem} [{f.parent.name}]{f.suffix}"
            shutil.copy(f, dest)
            count += 1
    print(f"Total de documentos validos copiados: {count}")


if __name__ == "__main__":
    download()
    filter_documents()
