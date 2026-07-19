"""Configurações centrais do projeto (variáveis de ambiente e constantes)."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = BASE_DIR / "data" / "CNMP_docs"
CHROMA_DIR = BASE_DIR / "data" / "chroma_db"
COLLECTION_NAME = "cnmp_docs"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

if not OPENROUTER_API_KEY:
    raise EnvironmentError(
        "OPENROUTER_API_KEY nao encontrada. Configure o arquivo .env (veja .env.example)."
    )
