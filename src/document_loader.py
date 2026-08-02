"""Extracao de texto de PDF, TXT e HTML, com fallback de OCR para PDFs escaneados."""
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup
from pdf2image import convert_from_path
import pytesseract


def _ocr_pdf(filepath: Path) -> str:
    try:
        paginas = convert_from_path(str(filepath), dpi=200)
        return "\n".join(pytesseract.image_to_string(p, lang="por") for p in paginas)
    except Exception as exc:
        print(f"[ERRO OCR] Falha ao processar {filepath.name}: {exc}")
        return ""


def extract_text(filepath: Path) -> str:
    """Extrai texto de PDF, TXT ou HTML. Usa OCR como fallback quando o PDF nao tem texto embutido."""
    try:
        ext = filepath.suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(str(filepath))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            if not text.strip():
                text = _ocr_pdf(filepath)
            return text
        elif ext == ".txt":
            return filepath.read_text(encoding="utf-8", errors="ignore")
        elif ext in (".html", ".htm"):
            html = filepath.read_text(encoding="utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            return soup.get_text(separator="\n")
        return ""
    except Exception as exc:
        print(f"[ERRO] Falha ao extrair {filepath.name}: {exc}")
        return ""


def load_documents(docs_dir: Path) -> list:
    """Carrega e extrai o texto de todos os documentos validos em docs_dir."""
    documents = []
    for f in sorted(docs_dir.iterdir()):
        if f.is_file():
            text = extract_text(f)
            if text.strip():
                documents.append({"source": f.name, "text": text})
            else:
                print(f"[AVISO] Documento vazio ou ilegivel: {f.name}")
    return documents
