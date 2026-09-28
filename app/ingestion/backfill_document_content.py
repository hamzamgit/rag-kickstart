"""Populate full document text and page counts for an existing database."""

from dotenv import load_dotenv

from app.ingestion.ingest_documents_to_postgres import (
    PROJECT_ROOT,
    discover_documents,
    extract_html_document,
    extract_pdf_document,
    get_database_connection,
)


def ensure_document_content_columns(cursor) -> None:
    """Make the backfill safe for databases created before these columns existed."""
    cursor.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS content TEXT;")
    cursor.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS page_count INTEGER;")


def backfill_document_content() -> int:
    """Store full extracted text without re-embedding or replacing chunks."""
    load_dotenv(PROJECT_ROOT / ".env")
    updated_count = 0

    with get_database_connection() as connection, connection.cursor() as cursor:
        ensure_document_content_columns(cursor)

        for document in discover_documents():
            if document["document_type"] == "pdf":
                full_content, page_count, _ = extract_pdf_document(
                    document["source_path"]
                )
            else:
                full_content, page_count, _ = extract_html_document(
                    document["source_path"]
                )

            cursor.execute(
                """
                UPDATE documents
                SET content = %s, page_count = %s
                WHERE source_filename = %s;
                """,
                (full_content, page_count, document["filename"]),
            )
            updated_count += cursor.rowcount

            print(
                f"Stored full content for {document['filename']} "
                f"({page_count} pages)."
            )

    print(f"Backfilled {updated_count} document records.")
    return updated_count


def backfill_documents_and_multi_vectors() -> tuple[int, int]:
    """Backfill full text, then index only chunks missing multi-vectors.

    This is the one-command recovery workflow for an existing database. It
    does not replace normal chunks or their embeddings. The multi-vector step
    skips every chunk that already has representation rows.
    """
    document_count = backfill_document_content()

    # Imported here to keep the basic content-backfill function independent of
    # the LLM-dependent multi-vector indexing step.
    from app.indexing.multivector.multi_vector_retrieval import (
        index_missing_chunk_representations,
    )

    multi_vector_chunk_count = index_missing_chunk_representations()
    return document_count, multi_vector_chunk_count


if __name__ == "__main__":
    document_count, multi_vector_chunk_count = backfill_documents_and_multi_vectors()
    print(
        "Completed backfill: "
        f"{document_count} documents updated; "
        f"{multi_vector_chunk_count} chunks received multi-vectors."
    )
