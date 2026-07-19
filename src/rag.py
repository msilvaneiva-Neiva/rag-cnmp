"""Pipeline RAG: recuperacao de contexto + geracao de resposta com o LLM (com fontes)."""
from openai import OpenAI
from . import config


class RAGEngine:
    def __init__(self, retriever):
        self.retriever = retriever
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config.OPENROUTER_API_KEY,
        )

    def answer(self, question: str, n_results: int = 10) -> dict:
        """Recupera contexto e gera resposta. Retorna dict com answer e sources."""
        try:
            retrieved = self.retriever.search(question, n_results=n_results)

            context = "\n\n---\n\n".join(
                f"[Fonte: {c['source']}]\n{c['text']}" for c in retrieved
            )

            prompt = f"""Voce e um assistente especializado em responder perguntas sobre o Premio CNMP com base em documentos oficiais.

Use APENAS as informacoes do contexto abaixo para responder a pergunta. Se a resposta nao estiver no contexto, diga que nao encontrou essa informacao nos documentos.

Contexto:
{context}

Pergunta: {question}

Resposta:"""

            resp = self.client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            answer = resp.choices[0].message.content
            sources = sorted(set(c["source"] for c in retrieved))
            return {"answer": answer, "sources": sources}

        except Exception as exc:
            return {"answer": f"[ERRO] Nao foi possivel gerar a resposta: {exc}", "sources": []}
