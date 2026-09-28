"""pgvector retrieval for documents loaded by the ingestion package."""

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


class PostgresRetriever:
    """Retrieve PDF chunks from PostgreSQL by cosine similarity."""

    def __init__(
        self,
        *,
        k: int = 3,
        topics: Sequence[str] | None = None,
        max_published_year: int | None = None,
    ):
        if k < 1:
            raise ValueError("k must be at least 1.")

        self.k = k
        self.topics = tuple(topics) if topics else None
        self.max_published_year = max_published_year
        self._embeddings = None

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Load the embedding model only when a query is actually executed."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return self._embeddings

    def invoke(
        self,
        query: str,
        *,
        topics: Sequence[str] | None = None,
        max_published_year: int | None = None,
    ) -> list[Document]:
        """Return the closest stored chunks, optionally constrained by metadata.

        Values passed to this method apply to this one search only. They are
        useful for examples such as query construction, where each user query
        can request different metadata filters.
        """
        if not query or not query.strip():
            return []

        embedding = Vector(self.embeddings.embed_query(query))

        selected_topics = tuple(topics) if topics is not None else self.topics
        selected_max_year = (
            max_published_year
            if max_published_year is not None
            else self.max_published_year
        )

        filters = []
        parameters = [embedding]
        if selected_topics:
            filters.append("d.topic = ANY(%s)")
            parameters.append(list(selected_topics))
        if selected_max_year is not None:
            filters.append("d.published_year <= %s")
            parameters.append(selected_max_year)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        query_sql = f"""
            SELECT c.content, c.metadata, d.title, d.source_filename,
                   d.source_url, d.topic, d.document_type,
                   d.published_year, c.chunk_index,
                   c.embedding <=> %s AS distance
            FROM document_chunks AS c
            JOIN documents AS d ON d.id = c.document_id
            {where_clause}
            ORDER BY c.embedding <=> %s
            LIMIT %s;
        """
        parameters.extend((embedding, self.k))

        with _get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(query_sql, parameters)

            rows = cursor.fetchall()

        return [
            Document(
                page_content=row[0],
                metadata={
                    **(row[1] or {}),
                    "title": row[2],
                    "source_filename": row[3],
                    "source_url": row[4],
                    "topic": row[5],
                    "document_type": row[6],
                    "published_year": row[7],
                    "chunk_index": row[8],
                    "distance": float(row[9]),
                },
            )
            for row in rows
        ]


def _get_database_connection():
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing from the project .env file.")

    connection = psycopg.connect(database_url)
    register_vector(connection)
    return connection


def build_retriever(
    *,
    k: int = 3,
    topics: Sequence[str] | None = None,
    max_published_year: int | None = None,
) -> PostgresRetriever:
    """Build a retriever over the PDF chunks already stored in PostgreSQL."""
    return PostgresRetriever(
        k=k,
        topics=topics,
        max_published_year=max_published_year,
    )
