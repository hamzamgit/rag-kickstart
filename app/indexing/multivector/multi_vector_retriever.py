"""Database access for the per-chunk multi-vector indexing workflow.

This module deliberately does not change ``app.rag.retriever``. The normal RAG
retriever continues to search one embedding per chunk. This retriever searches
the dedicated ``chunk_multi_vectors`` table, where one chunk can have many
representations and therefore many embeddings.
"""

import os
from collections.abc import Sequence
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pgvector import Vector
from pgvector.psycopg import register_vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


MULTI_VECTOR_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunk_multi_vectors (
    id BIGSERIAL PRIMARY KEY,
    chunk_id BIGINT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    representation_type TEXT NOT NULL,
    representation_index SMALLINT NOT NULL DEFAULT 0,
    representation_text TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_chunk_representation
        UNIQUE (chunk_id, representation_type, representation_index)
);

CREATE INDEX IF NOT EXISTS chunk_multi_vectors_chunk_id_idx
    ON chunk_multi_vectors (chunk_id);

CREATE INDEX IF NOT EXISTS chunk_multi_vectors_embedding_hnsw_idx
    ON chunk_multi_vectors
    USING hnsw (embedding vector_cosine_ops);
"""


def get_database_connection():
    """Open a pgvector-ready connection without using the normal retriever."""
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is missing from the project .env file.")

    connection = psycopg.connect(database_url)
    register_vector(connection)
    return connection


class MultiVectorPostgresRetriever:
    """Search multi-vectors, then return a full document or chunk by page count."""

    def __init__(self, *, k: int = 3, full_document_page_limit: int = 40):
        if k < 1:
            raise ValueError("k must be at least 1.")
        if full_document_page_limit < 1:
            raise ValueError("full_document_page_limit must be at least 1.")

        self.k = k
        self.full_document_page_limit = full_document_page_limit
        self._embeddings = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Load the embedding model only when indexing or searching begins."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return self._embeddings

    def ensure_schema(self) -> None:
        """Create the separate table used only by this learning example."""
        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(MULTI_VECTOR_SCHEMA_SQL)

    def chunks_without_representations(self, *, limit: int | None = None) -> list[dict]:
        """Return stored chunks that have not yet received their extra vectors."""
        query_sql = """
            SELECT c.id, c.content, d.title, d.source_filename, d.topic
            FROM document_chunks AS c
            JOIN documents AS d ON d.id = c.document_id
            WHERE NOT EXISTS (
                SELECT 1
                FROM chunk_multi_vectors AS mv
                WHERE mv.chunk_id = c.id
            )
            ORDER BY c.id
        """
        parameters = []

        if limit is not None:
            query_sql += " LIMIT %s"
            parameters.append(limit)

        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(query_sql, parameters)
            rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "content": row[1],
                "title": row[2],
                "source_filename": row[3],
                "topic": row[4],
            }
            for row in rows
        ]

    def replace_representations(
        self,
        chunk_id: int,
        representations: Sequence[tuple[str, int, str]],
    ) -> None:
        """Replace every vector representation belonging to one parent chunk."""
        texts = [representation_text for _, _, representation_text in representations]
        vectors = self.embeddings.embed_documents(texts)

        rows = [
            (
                chunk_id,
                representation_type,
                representation_index,
                representation_text,
                Vector(vector),
            )
            for (representation_type, representation_index, representation_text), vector
            in zip(representations, vectors)
        ]

        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chunk_multi_vectors WHERE chunk_id = %s;",
                (chunk_id,),
            )
            cursor.executemany(
                """
                INSERT INTO chunk_multi_vectors (
                    chunk_id,
                    representation_type,
                    representation_index,
                    representation_text,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s);
                """,
                rows,
            )

    def invoke(self, query: str) -> list[Document]:
        """Search representations, then choose full-document or chunk context.

        A chunk already has ``document_id``. For short documents, that ID lets
        us fetch ``documents.content`` as the LLM context. Longer documents
        return only their best matching chunk to avoid overfilling the model's
        context window.
        """
        if not query or not query.strip():
            return []

        query_vector = Vector(self.embeddings.embed_query(query))

        query_sql = """
            WITH closest_representation_per_chunk AS (
                SELECT DISTINCT ON (mv.chunk_id)
                    mv.chunk_id,
                    mv.representation_type,
                    mv.representation_index,
                    mv.embedding <=> %s AS distance
                FROM chunk_multi_vectors AS mv
                ORDER BY mv.chunk_id, mv.embedding <=> %s
            ),
            closest_chunk_per_document AS (
                SELECT DISTINCT ON (c.document_id)
                    c.content,
                    c.metadata,
                    d.title,
                    d.source_filename,
                    d.topic,
                    d.id AS document_id,
                    d.content AS document_content,
                    d.page_count,
                    c.chunk_index,
                    closest.representation_type,
                    closest.representation_index,
                    closest.distance
                FROM closest_representation_per_chunk AS closest
                JOIN document_chunks AS c ON c.id = closest.chunk_id
                JOIN documents AS d ON d.id = c.document_id
                ORDER BY c.document_id, closest.distance
            )
            SELECT
                content,
                metadata,
                title,
                source_filename,
                topic,
                document_id,
                document_content,
                page_count,
                chunk_index,
                representation_type,
                representation_index,
                distance
            FROM closest_chunk_per_document
            ORDER BY distance
            LIMIT %s;
        """

        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(query_sql, (query_vector, query_vector, self.k))
            rows = cursor.fetchall()

        documents = []

        for row in rows:
            chunk_content = row[0]
            document_content = row[6]
            page_count = row[7]
            use_full_document = (
                document_content is not None
                and page_count is not None
                and page_count <= self.full_document_page_limit
            )

            documents.append(
                Document(
                    page_content=document_content if use_full_document else chunk_content,
                    metadata={
                        **(row[1] or {}),
                        "title": row[2],
                        "source_filename": row[3],
                        "topic": row[4],
                        "document_id": str(row[5]),
                        "page_count": page_count,
                        "chunk_index": row[8],
                        "retrieval_scope": "full_document" if use_full_document else "chunk",
                        "matched_representation_type": row[9],
                        "matched_representation_index": row[10],
                        "distance": float(row[11]),
                    },
                )
            )

        return documents


def build_multi_vector_retriever(
    *,
    k: int = 3,
    full_document_page_limit: int = 40,
) -> MultiVectorPostgresRetriever:
    """Build the separate retriever used by real per-chunk multi-vector RAG."""
    return MultiVectorPostgresRetriever(
        k=k,
        full_document_page_limit=full_document_page_limit,
    )
