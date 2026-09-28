# Cohere LangGraph Adaptive RAG

This is the graph-based version of the Cohere Adaptive RAG workflow. The
non-graph direct implementation remains at `../adaptive_rag.py`.

```text
START -> route
  |- database-search -> retrieve -> grade each source
  |      |- relevant -> generate -> grounding + usefulness -> answer
  |      `- no relevant source -> Tavily web search -> generate
  |- web-search -> Tavily web search -> generate
  `- unsupported route -> safe fallback

generation not grounded -> regenerate from the same evidence (bounded)
grounded but not useful -> web search -> generate (bounded)
```

Run it with the same `COHERE_API_KEY` and `TAVILY_API_KEY` from `../../../../.env`:

```bash
python -m app.agentic_rag.AdaptiveRAG.langgraph.adaptive_rag_graph "How does Self-RAG decide to retrieve?"
```

`--max-retries` defaults to `2`. The `@traceable` entry function creates one
top-level LangSmith trace per terminal run; graph nodes and Cohere calls appear
as child runs.
