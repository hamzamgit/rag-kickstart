"""Separate pgvector collection for RAPTOR tree nodes."""

import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pgvector import Vector
from pgvector.psycopg import register_vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


RAPTOR_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS raptor_nodes (
    id BIGSERIAL PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    source_chunk_id BIGINT REFERENCES document_chunks(id) ON DELETE CASCADE,
    parent_node_id BIGINT REFERENCES raptor_nodes(id) ON DELETE CASCADE,
    level SMALLINT NOT NULL CHECK (level >= 0),
    node_type TEXT NOT NULL CHECK (node_type IN ('leaf', 'summary', 'root')),
    cluster_index INTEGER,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_raptor_leaf UNIQUE (source_chunk_id),
    CONSTRAINT unique_raptor_node UNIQUE (document_id, level, node_type, cluster_index)
);

CREATE INDEX IF NOT EXISTS raptor_nodes_document_id_idx
    ON raptor_nodes (document_id);

CREATE INDEX IF NOT EXISTS raptor_nodes_embedding_hnsw_idx
    ON raptor_nodes
    USING hnsw (embedding vector_cosine_ops);
"""


def get_database_connection():
    """Open a pgvector-ready connection dedicated to the RAPTOR collection."""
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing from the project .env file.")

    connection = psycopg.connect(database_url)
    register_vector(connection)
    return connection


def normalize_embedding(embedding) -> np.ndarray:
    """Convert pgvector's read type into an ndarray accepted by ``Vector``.

    ``register_vector`` returns a ``pgvector.Vector`` when reading an embedding
    from PostgreSQL. That object is not itself valid input to ``Vector(...)``;
    converting it to NumPy also gives GMM a numeric matrix for clustering.
    """
    if isinstance(embedding, Vector):
        return embedding.to_numpy()

    return np.asarray(embedding, dtype=np.float32)


class RaptorPostgresRetriever:
    """Build and search a flat pool containing RAPTOR leaves and summaries."""

    def __init__(self, *, k: int = 8):
        if k < 1:
            raise ValueError("k must be at least 1.")

        self.k = k
        self._embeddings = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Load the embedding model only when summaries need embedding or search."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return self._embeddings

    def ensure_schema(self) -> None:
        """Create the separate table used only by RAPTOR."""
        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(RAPTOR_SCHEMA_SQL)

    def documents_without_root(self) -> list[dict]:
        """Return documents with no completed RAPTOR tree yet."""
        query_sql = """
            SELECT d.id, d.title, d.source_filename, d.topic
            FROM documents AS d
            WHERE NOT EXISTS (
                SELECT 1
                FROM raptor_nodes AS node
                WHERE node.document_id = d.id AND node.node_type = 'root'
            )
            ORDER BY d.source_filename;
        """
        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(query_sql)
            rows = cursor.fetchall()

        return [
            {
                "document_id": row[0],
                "title": row[1],
                "source_filename": row[2],
                "topic": row[3],
            }
            for row in rows
        ]

    def get_document_chunks(self, document_id) -> list[dict]:
        """Read original chunks and their existing vectors for RAPTOR leaves."""
        query_sql = """
            SELECT id, chunk_index, content, embedding
            FROM document_chunks
            WHERE document_id = %s
            ORDER BY chunk_index;
        """
        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(query_sql, (document_id,))
            rows = cursor.fetchall()

        return [
            {
                "chunk_id": row[0],
                "chunk_index": row[1],
                "content": row[2],
                "embedding": normalize_embedding(row[3]),
            }
            for row in rows
        ]

    def clear_document_tree(self, document_id) -> None:
        """Remove an incomplete or outdated tree before rebuilding that document."""
        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute("DELETE FROM raptor_nodes WHERE document_id = %s;", (document_id,))

    def insert_leaf_nodes(self, document_id, chunks: list[dict]) -> list[dict]:
        """Copy normal chunk vectors into the RAPTOR collection as level-zero leaves."""
        rows = [
            (
                document_id,
                chunk["chunk_id"],
                chunk["chunk_index"],
                chunk["content"],
                Vector(chunk["embedding"]),
            )
            for chunk in chunks
        ]

        with get_database_connection() as connection, connection.cursor() as cursor:
            node_ids = []
            for row in rows:
                cursor.execute(
                    """
                    INSERT INTO raptor_nodes (
                        document_id, source_chunk_id, level, node_type,
                        cluster_index, content, embedding
                    )
                    VALUES (%s, %s, 0, 'leaf', %s, %s, %s)
                    RETURNING id;
                    """,
                    row,
                )
                node_ids.append(cursor.fetchone()[0])

        return [
            {
                "node_id": node_id,
                "content": chunk["content"],
                "embedding": chunk["embedding"],
            }
            for node_id, chunk in zip(node_ids, chunks)
        ]

    def insert_summary_node(
        self,
        *,
        document_id,
        parent_node_id: int | None,
        level: int,
        node_type: str,
        cluster_index: int,
        content: str,
        embedding,
        child_node_ids: list[int],
    ) -> dict:
        """Store one summary/root node and return it for the next tree level."""
        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO raptor_nodes (
                    document_id, parent_node_id, level, node_type,
                    cluster_index, content, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """,
                (
                    document_id,
                    parent_node_id,
                    level,
                    node_type,
                    cluster_index,
                    content,
                    Vector(embedding),
                ),
            )
            node_id = cursor.fetchone()[0]
            cursor.execute(
                "UPDATE raptor_nodes SET parent_node_id = %s WHERE id = ANY(%s);",
                (node_id, child_node_ids),
            )

        return {"node_id": node_id, "content": content, "embedding": embedding}

    def invoke(self, query: str) -> list[Document]:
        """Run one normal top-k search over leaves, summaries, and roots together."""
        if not query or not query.strip():
            return []

        query_vector = Vector(self.embeddings.embed_query(query))
        query_sql = """
            SELECT
                node.content,
                node.level,
                node.node_type,
                node.cluster_index,
                node.document_id,
                d.title,
                d.source_filename,
                d.topic,
                node.embedding <=> %s AS distance
            FROM raptor_nodes AS node
            JOIN documents AS d ON d.id = node.document_id
            ORDER BY node.embedding <=> %s
            LIMIT %s;
        """
        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(query_sql, (query_vector, query_vector, self.k))
            rows = cursor.fetchall()

        return [
            Document(
                page_content=row[0],
                metadata={
                    "level": row[1],
                    "node_type": row[2],
                    "cluster_index": row[3],
                    "document_id": str(row[4]),
                    "title": row[5],
                    "source_filename": row[6],
                    "topic": row[7],
                    "distance": float(row[8]),
                },
            )
            for row in rows
        ]


def build_raptor_retriever(*, k: int = 8) -> RaptorPostgresRetriever:
    """Build the separate retriever for flat RAPTOR node search."""
    return RaptorPostgresRetriever(k=k)
