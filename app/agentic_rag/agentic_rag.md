# Agentic RAG

These examples add explicit decisions to ordinary retrieve-then-answer RAG.
They use the existing PostgreSQL corpus and `app.rag.retriever`; neither
creates a separate index.

| Technique | Learning guide | Main decision |
| --- | --- | --- |
| Corrective RAG (CRAG) | [CRAG guide](CRAG/crag.md) | Are the retrieved chunks relevant enough to answer? |
| Adaptive RAG | [Adaptive RAG guide](AdaptiveRAG/adaptive_rag.md) | Which source and recovery path should answer this question? |

Run either example after document ingestion and PostgreSQL are available. Both
plain-Python and LangGraph implementations are provided in their respective
directories.
