"""Topic-scoped pgvector retrievers used by the routing RAG example."""

from app.rag.retriever import build_retriever


# These values match the topic metadata currently stored by PDF ingestion.
AGENT_TOPICS = ("agents",)
RAG_TOPICS = ("rag", "hyde", "self_rag")


def build_routing_retrievers():
    """Return retrievers that query the relevant documents in PostgreSQL."""
    return {
        "agent": build_retriever(k=3, topics=AGENT_TOPICS),
        "rag": build_retriever(k=3, topics=RAG_TOPICS),
    }
