"""Segmentacao (chunking) dos textos extraidos."""
from langchain_text_splitters import RecursiveCharacterTextSplitter
from . import config


def chunk_documents(documents: list) -> list:
    """Divide os documentos em chunks menores, preservando metadados de origem."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(doc_chunks):
            chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "chunk_id": f"{doc['source']}__chunk{i}",
            })
    return chunks
