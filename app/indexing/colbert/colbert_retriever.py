"""Official ColBERTv2 late-interaction retrieval over local database chunks."""

import csv
import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLBERT_CHECKPOINT = "colbert-ir/colbertv2.0"
COLBERT_DATA_DIRECTORY = PROJECT_ROOT / "data" / "colbert"
COLLECTION_PATH = COLBERT_DATA_DIRECTORY / "collections" / "local_chunks.tsv"
PASSAGE_ID_MAP_PATH = COLBERT_DATA_DIRECTORY / "collections" / "local_chunks_ids.json"
INDEX_ROOT = COLBERT_DATA_DIRECTORY / "indexes"
INDEX_NAME = "local_chunks"


def get_database_connection():
    """Open a connection used only to export/map ColBERT passage IDs."""
    load_dotenv(PROJECT_ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is missing from the project .env file.")
    return psycopg.connect(database_url)


def require_colbert():
    """Load the official package only when indexing/searching is requested."""
    try:
        from colbert import Indexer, Searcher
        from colbert.infra import ColBERTConfig, Run, RunConfig
    except ImportError as error:
        raise RuntimeError(
            "ColBERT is not installed. Install it in a supported GPU Python "
            "environment: pip install 'colbert-ai[torch,faiss-gpu]'"
        ) from error
    return Indexer, Searcher, ColBERTConfig, Run, RunConfig


def export_postgres_chunks() -> int:
    """Export zero-based ColBERT passage IDs and save their database-ID mapping."""
    COLLECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_database_connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT id, content FROM document_chunks ORDER BY id;")
        rows = cursor.fetchall()

    with COLLECTION_PATH.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.writer(file_handle, delimiter="\t", lineterminator="\n")
        passage_to_chunk_id = {}
        for passage_id, (chunk_id, content) in enumerate(rows):
            # ColBERT 0.2.x requires a TSV passage ID to equal its line number.
            writer.writerow((passage_id, " ".join(content.split())))
            passage_to_chunk_id[passage_id] = chunk_id

    with PASSAGE_ID_MAP_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(passage_to_chunk_id, file_handle)

    print(f"Exported {len(rows)} chunks to {COLLECTION_PATH}.")
    return len(rows)


def build_colbert_index() -> None:
    """Create an official ColBERTv2 token index without any LLM call."""
    passage_count = export_postgres_chunks()
    if not passage_count:
        raise ValueError("No document chunks are available to index.")

    Indexer, _, ColBERTConfig, Run, RunConfig = require_colbert()
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    with Run().context(RunConfig(nranks=1, experiment="local_colbert")):
        config = ColBERTConfig(nbits=2, root=str(INDEX_ROOT))
        indexer = Indexer(checkpoint=COLBERT_CHECKPOINT, config=config)
        indexer.index(
            name=INDEX_NAME,
            collection=str(COLLECTION_PATH),
            overwrite=True,
        )

    print(f"Created ColBERT index: {INDEX_ROOT / INDEX_NAME}")


class ColBERTRetriever:
    """Search with ColBERT MaxSim, then map passage IDs to PostgreSQL chunks."""

    def __init__(self, *, k: int = 5):
        if k < 1:
            raise ValueError("k must be at least 1.")
        self.k = k

    def invoke(self, query: str) -> list[Document]:
        """Return original chunks found by token-level late interaction."""
        if not query or not query.strip():
            return []

        _, Searcher, ColBERTConfig, Run, RunConfig = require_colbert()
        with Run().context(RunConfig(nranks=1, experiment="local_colbert")):
            config = ColBERTConfig(root=str(INDEX_ROOT))
            searcher = Searcher(index=INDEX_NAME, config=config)
            passage_ids, ranks, scores = searcher.search(query, k=self.k)

        rankings = [
            (int(passage_id), int(rank), float(score))
            for passage_id, rank, score in zip(passage_ids, ranks, scores)
        ]
        if not rankings:
            return []

        try:
            with PASSAGE_ID_MAP_PATH.open(encoding="utf-8") as file_handle:
                passage_to_chunk_id = {
                    int(passage_id): int(chunk_id)
                    for passage_id, chunk_id in json.load(file_handle).items()
                }
        except FileNotFoundError as error:
            raise RuntimeError(
                "The ColBERT passage-ID map is missing. Rebuild the index with "
                "`python -m app.indexing.colbert --build-index`."
            ) from error

        chunk_ids = [
            passage_to_chunk_id[passage_id]
            for passage_id, _, _ in rankings
            if passage_id in passage_to_chunk_id
        ]
        with get_database_connection() as connection, connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT c.id, c.content, c.chunk_index, c.metadata,
                       d.id, d.title, d.source_filename, d.topic
                FROM document_chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE c.id = ANY(%s);
                """,
                (chunk_ids,),
            )
            rows = cursor.fetchall()

        chunks_by_id = {row[0]: row for row in rows}
        documents = []
        for passage_id, rank, score in rankings:
            chunk_id = passage_to_chunk_id.get(passage_id)
            row = chunks_by_id.get(chunk_id)
            if row is None:
                continue
            documents.append(
                Document(
                    page_content=row[1],
                    metadata={
                        **(row[3] or {}),
                        "chunk_id": row[0],
                        "chunk_index": row[2],
                        "document_id": str(row[4]),
                        "title": row[5],
                        "source_filename": row[6],
                        "topic": row[7],
                        "colbert_rank": rank,
                        "colbert_score": score,
                    },
                )
            )
        return documents


def build_colbert_retriever(*, k: int = 5) -> ColBERTRetriever:
    """Build the separate retriever that uses official ColBERT late interaction."""
    return ColBERTRetriever(k=k)
