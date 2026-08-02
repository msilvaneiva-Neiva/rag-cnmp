"""Ponto de entrada: monta o pipeline completo e sobe a interface Gradio."""
import gradio as gr
from src import config
from src.document_loader import load_documents
from src.chunking import chunk_documents
from src.vectorstore import get_collection, index_chunks
from src.retrieval import HybridRetriever
from src.rag import RAGEngine


def build_pipeline():
    print("Carregando documentos...")
    documents = load_documents(config.DOCS_DIR)
    print(f"{len(documents)} documentos carregados.")

    print("Gerando chunks...")
    chunks = chunk_documents(documents)
    print(f"{len(chunks)} chunks gerados.")

    print("Indexando no ChromaDB...")
    collection = get_collection()
    if collection.count() == 0:
        index_chunks(collection, chunks)
    else:
        print(f"Colecao ja indexada ({collection.count()} itens). Pulando reindexacao.")

    retriever = HybridRetriever(collection, chunks)
    return RAGEngine(retriever)


def main():
    engine = build_pipeline()

    def rag_interface(question):
        if not question or not question.strip():
            return "Por favor, digite uma pergunta.", ""
        result = engine.answer(question)
        sources_text = "\n".join(f"- {s}" for s in result["sources"]) if result["sources"] else "Nenhuma fonte recuperada."
        return result["answer"], sources_text

    demo = gr.Interface(
        fn=rag_interface,
        inputs=gr.Textbox(label="Pergunta", placeholder="Ex: Quem venceu a categoria Tecnologia da Informacao em 2013?"),
        outputs=[
            gr.Textbox(label="Resposta", lines=8),
            gr.Textbox(label="Fontes utilizadas", lines=6),
        ],
        title="Assistente CNMP - Consulta ao Premio CNMP (RAG)",
        description="Sistema de perguntas e respostas baseado em RAG sobre os documentos do Premio CNMP (2013-2026).",
    )
    demo.launch(share=True)


if __name__ == "__main__":
    main()
