# Indexing and query flow

PostgreSQL is the source of truth for documents and chunks. Each advanced
retrieval technique creates a separate index or collection.

## Shared ingestion

```text
PDF or HTML
  -> full extracted text -> documents.content
  -> split text          -> document_chunks.content
  -> normal embedding    -> document_chunks.embedding
```

`documents.id` identifies a source document. Every `document_chunks` row has a
`document_id` linking the chunk to its parent document.

## Multi-vector retrieval

```text
one original chunk
  -> original-text vector
  -> summary vector
  -> key-concepts vector
  -> five hypothetical-question vectors
  -> chunk_multi_vectors
```

One query vector searches all eight representations. The winning representation
maps to `chunk_id`, then to the original chunk or parent document. Summaries,
concepts, and hypothetical questions improve matching only; they are not final
answer context.

## RAPTOR retrieval

```text
original chunks (leaf nodes)
  -> GMM clusters similar vectors
  -> LLM summarizes each cluster
  -> summary nodes
  -> repeat until one root summary
  -> raptor_nodes
```

One query vector searches the flat pool of leaf, summary, and root nodes. The
final LLM receives the returned chunks and summaries together.

## ColBERT late-interaction retrieval

```text
document_chunks in PostgreSQL
  -> export: chunk_id<TAB>chunk text
  -> ColBERTv2 tokenizes every chunk
  -> contextual embedding matrix per chunk
  -> compressed official on-disk ColBERT index
```

No LLM call happens during ColBERT export or indexing.

At query time:

```text
user question
  -> contextual embedding matrix for query tokens
  -> MaxSim: each query token finds its best passage-token match
  -> sum best token matches into a passage score
  -> top-k passage IDs (PostgreSQL chunk IDs)
  -> fetch original chunks and metadata from PostgreSQL
  -> final LLM answer using only original chunks
```

ColBERT preserves token-level detail until scoring. Its standard compressed
index is stored on disk because it is searched differently from pgvector's
single-vector rows.
