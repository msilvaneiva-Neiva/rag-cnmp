"""Recuperacao de contexto: busca hibrida (vetorial + lexical/BM25) - inclui funcionalidade bonus."""
import re
from rank_bm25 import BM25Okapi

WINNER_KEYWORDS = [
    "venceu", "vencedor", "vencedora", "ganhou", "premiacao", "premiacoes",
    "premiado", "premiada", "1 lugar", "2 lugar", "3 lugar", "resultado",
]


def tokenize(text: str) -> list:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """Combina busca vetorial (ChromaDB) e lexical (BM25) via Reciprocal Rank Fusion,
    com priorizacao de arquivos Premiados para perguntas sobre vencedores."""

    def __init__(self, collection, chunks: list):
        self.collection = collection
        self.chunks = chunks
        self.chunk_lookup = {c["chunk_id"]: c for c in chunks}
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, n_results: int = 10, vector_k: int = 20, bm25_k: int = 20) -> list:
        vec_results = self.collection.query(query_texts=[query], n_results=vector_k)
        vec_ids = vec_results["ids"][0]

        bm25_scores = self.bm25.get_scores(tokenize(query))
        bm25_top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_k]
        bm25_ids = [self.chunks[i]["chunk_id"] for i in bm25_top_idx]

        scores = {}
        for rank, cid in enumerate(vec_ids):
            scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)
        for rank, cid in enumerate(bm25_ids):
            scores[cid] = scores.get(cid, 0) + 1 / (60 + rank)

        ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:n_results]
        result_chunks = [self.chunk_lookup[cid] for cid in ranked_ids if cid in self.chunk_lookup]

        is_winner_query = any(kw in query.lower() for kw in WINNER_KEYWORDS)
        if is_winner_query:
            result_chunks = self._boost_premiados(query, result_chunks)

        return result_chunks

    def _boost_premiados(self, query: str, result_chunks: list) -> list:
        entity_candidates = re.findall(r"MP[/\s]?[A-Z]{2,3}\b", query, flags=re.IGNORECASE)
        entity_candidates += re.findall(r"\b[A-ZA-Z][a-z]{3,}\b", query)

        premiados_chunks = [c for c in self.chunks if "Premiados" in c["source"]]
        existing_ids = {c["chunk_id"] for c in result_chunks}

        for entity in entity_candidates:
            entity_norm = entity.replace(" ", "").lower()
            for c in premiados_chunks:
                if entity_norm in c["text"].replace(" ", "").lower() and c["chunk_id"] not in existing_ids:
                    result_chunks.append(c)
                    existing_ids.add(c["chunk_id"])
        return result_chunks
