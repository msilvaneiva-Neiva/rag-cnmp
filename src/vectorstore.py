"""Geracao de embeddings e armazenamento no banco vetorial ChromaDB."""
import chromadb
from chromadb.utils import embedding_functions
from . import config


def get_collection():
    """Cria/conecta a colecao ChromaDB persistente."""
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def index_chunks(collection, chunks: list, batch_size: int = 100) -> None:
    """Insere os chunks na colecao em lotes."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch],
        )
    print(f"Total de itens indexados: {collection.count()}")
