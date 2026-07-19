"""Extracao de texto de documentos PDF, TXT e HTML (Coleta e Preparacao dos Dados)."""
from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup


def extract_text(filepath: Path) -> str:
    """Extrai texto de um arquivo PDF, TXT ou HTML. Retorna string vazia em caso de erro."""
    try:
        ext = filepath.suffix.lower()
        if ext == ".pdf":
            reader = PdfReader(str(filepath))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
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
