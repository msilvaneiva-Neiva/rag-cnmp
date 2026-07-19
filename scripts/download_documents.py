"""Script para baixar e filtrar a base documental do Google Drive."""
import shutil
import subprocess
from pathlib import Path

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1mdJJ-a1nXe4wSEMUUI74b5af07x_W-ch"
RAW_DIR = Path("data/CNMP_raw")
DOCS_DIR = Path("data/CNMP_docs")
VALID_EXT = {".pdf", ".txt", ".html", ".htm"}


def download():
    subprocess.run(["gdown", "--folder", DRIVE_FOLDER_URL, "-O", str(RAW_DIR), "--remaining-ok"], check=False)


def filter_documents():
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for f in RAW_DIR.rglob("*"):
        if f.is_file() and f.suffix.lower() in VALID_EXT:
            dest_name = f"{f.parent.name}__{f.name}" if f.parent != RAW_DIR else f.name
            shutil.copy(f, DOCS_DIR / dest_name)
            count += 1
    print(f"Total de documentos validos copiados: {count}")


if __name__ == "__main__":
    download()
    filter_documents()
