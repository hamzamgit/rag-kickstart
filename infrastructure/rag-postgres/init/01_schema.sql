CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source_filename TEXT NOT NULL UNIQUE,
    source_url TEXT,
    topic TEXT NOT NULL,
    document_type TEXT NOT NULL,
    published_year SMALLINT,
    content TEXT,
    page_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_document_chunk UNIQUE (document_id, chunk_index)
);

CREATE INDEX documents_topic_year_idx
    ON documents (topic, published_year);

CREATE INDEX document_chunks_document_id_idx
    ON document_chunks (document_id);

CREATE INDEX document_chunks_embedding_hnsw_idx
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops);
