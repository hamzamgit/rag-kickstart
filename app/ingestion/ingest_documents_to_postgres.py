import os
import re
from html.parser import HTMLParser
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb
from pypdf import PdfReader

from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PDF_DIRECTORY = PROJECT_ROOT / "data" / "raw_pdfs"
HTML_DIRECTORY = PROJECT_ROOT / "data" / "raw_html"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 32

# Keep topic names stable when the source file names are more descriptive than
# the routes used by the RAG examples.
TOPIC_ALIASES = {
    "rag_original": "rag",
    "react_agents": "agents",
}


class _HTMLTextExtractor(HTMLParser):
    """Extract readable text while ignoring executable and styling content."""

    ignored_tags = {"script", "style", "noscript", "template"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.ignored_tags:
            self._ignored_depth += 1

    def handle_endtag(self, tag):
        if tag.lower() in self.ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data):
        if not self._ignored_depth:
            self.parts.append(data)


def filename_to_title(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"^\d+[_\-\s]*", "", stem)
    return stem.replace("_", " ").replace("-", " ").title()


def filename_to_topic(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"^\d+[_\-\s]*", "", stem)
    topic = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return TOPIC_ALIASES.get(topic, topic)


def discover_documents() -> list[dict]:
    source_files = [
        *( (path, "pdf") for path in sorted(PDF_DIRECTORY.glob("*.pdf")) ),
        *( (path, "html") for path in sorted(HTML_DIRECTORY.glob("*.html")) ),
        *( (path, "html") for path in sorted(HTML_DIRECTORY.glob("*.htm")) ),
    ]

    if not source_files:
        raise FileNotFoundError(
            "No PDF or HTML files found in "
            f"{PDF_DIRECTORY} or {HTML_DIRECTORY}"
        )

    documents = []

    for source_path, document_type in source_files:
        year_match = re.search(r"\b(19|20)\d{2}\b", source_path.stem)

        documents.append(
            {
                "filename": source_path.name,
                "source_path": source_path,
                "title": filename_to_title(source_path.name),
                "source_url": None,
                "topic": filename_to_topic(source_path.name),
                "document_type": document_type,
                "published_year": int(year_match.group()) if year_match else None,
            }
        )

    return documents

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        if end < len(text):
            last_space = text.rfind(" ", start + (CHUNK_SIZE // 2), end)
            if last_space != -1:
                end = last_space

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return chunks


def extract_pdf_document(pdf_path: Path) -> tuple[str, int, list[tuple[str, dict]]]:
    """Extract complete PDF text and its page-aware chunks in one pass."""
    reader = PdfReader(str(pdf_path))
    chunks = []
    page_texts = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = clean_text(page.extract_text() or "")

        if not page_text:
            continue

        page_texts.append(page_text)

        for chunk in split_text(page_text):
            chunks.append(
                (
                    chunk,
                    {
                        "page_number": page_number,
                        "source_filename": pdf_path.name,
                    },
                )
            )

    return "\n\n".join(page_texts), len(reader.pages), chunks


def extract_html_document(html_path: Path) -> tuple[str, int, list[tuple[str, dict]]]:
    """Extract complete HTML text and its chunks; HTML counts as one page."""
    extractor = _HTMLTextExtractor()
    extractor.feed(html_path.read_text(encoding="utf-8", errors="replace"))
    extractor.close()

    text = clean_text(" ".join(extractor.parts))
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    chunks = [
        (
            chunk,
            {
                "source_filename": html_path.name,
            },
        )
        for chunk in split_text(text)
    ]
    return text, 1, chunks


def get_database_connection():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is missing from the project .env file.")

    connection = psycopg.connect(database_url)
    register_vector(connection)

    return connection


def upsert_document(cursor, document: dict):
    cursor.execute(
        """
        INSERT INTO documents (
            title,
            source_filename,
            source_url,
            topic,
            document_type,
            published_year,
            content,
            page_count
        )
        VALUES (%(title)s, %(filename)s, %(source_url)s, %(topic)s,
                %(document_type)s, %(published_year)s, %(content)s, %(page_count)s)
        ON CONFLICT (source_filename)
        DO UPDATE SET
            title = EXCLUDED.title,
            source_url = EXCLUDED.source_url,
            topic = EXCLUDED.topic,
            document_type = EXCLUDED.document_type,
            published_year = EXCLUDED.published_year,
            content = EXCLUDED.content,
            page_count = EXCLUDED.page_count
        RETURNING id;
        """,
        document,
    )

    return cursor.fetchone()[0]


def ingest_document(connection, embeddings, document: dict):
    source_path = document["source_path"]

    if not source_path.exists():
        raise FileNotFoundError(f"Document not found: {source_path}")

    if document["document_type"] == "pdf":
        full_document_content, page_count, extracted_chunks = extract_pdf_document(
            source_path
        )
    elif document["document_type"] == "html":
        full_document_content, page_count, extracted_chunks = extract_html_document(
            source_path
        )
    else:
        raise ValueError(f"Unsupported document type: {document['document_type']}")

    if not extracted_chunks:
        raise ValueError(f"No extractable text found in: {source_path.name}")

    texts = [chunk_text for chunk_text, _ in extracted_chunks]

    print(f"\nEmbedding {source_path.name}: {len(texts)} chunks")

    vectors = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        vectors.extend(embeddings.embed_documents(batch))

    with connection.cursor() as cursor:
        document_to_store = {
            **document,
            "content": full_document_content,
            "page_count": page_count,
        }
        document_id = upsert_document(cursor, document_to_store)

        # Makes re-running the script idempotent: old chunks are replaced.
        cursor.execute(
            "DELETE FROM document_chunks WHERE document_id = %s;",
            (document_id,),
        )

        rows = []

        for chunk_index, ((chunk_text, metadata), vector) in enumerate(
            zip(extracted_chunks, vectors)
        ):
            metadata.update(
                {
                    "topic": document["topic"],
                    "published_year": document["published_year"],
                    "document_type": document["document_type"],
                }
            )

            rows.append(
                (
                    document_id,
                    chunk_index,
                    chunk_text,
                    Jsonb(metadata),
                    vector,
                )
            )

        cursor.executemany(
            """
            INSERT INTO document_chunks (
                document_id,
                chunk_index,
                content,
                metadata,
                embedding
            )
            VALUES (%s, %s, %s, %s, %s);
            """,
            rows,
        )

    connection.commit()

    print(f"Stored {len(rows)} chunks for {document['title']}")


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    with get_database_connection() as connection:
        for document in discover_documents():
            ingest_document(
                connection=connection,
                embeddings=embeddings,
                document=document,
            )

    # Complete the third storage step after documents and normal chunk vectors
    # have been committed. This function indexes only chunks with no existing
    # multi-vector rows, so previously indexed chunks are not processed again.
    from app.indexing.multivector.multi_vector_retrieval import (
        index_missing_chunk_representations,
    )

    multi_vector_chunk_count = index_missing_chunk_representations()

    print("\nIngestion completed successfully.")
    print(f"Multi-vector representations created for {multi_vector_chunk_count} chunks.")


if __name__ == "__main__":
    main()
