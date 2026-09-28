"""Small building blocks shared by CRAG and Adaptive RAG."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


def invoke_text(llm, template: str, **values: str) -> str:
    """Run a prompt and return normalized plain text."""
    prompt = ChatPromptTemplate.from_template(template)
    return (prompt | llm | StrOutputParser()).invoke(values).strip()


def format_documents(documents: list[Document], character_limit: int = 14_000) -> str:
    """Build bounded, source-labelled context for a final answer prompt."""
    sections = []
    remaining = character_limit
    for number, document in enumerate(documents, start=1):
        source = document.metadata.get("source_filename", "unknown source")
        chunk = document.metadata.get("chunk_index", "?")
        heading = f"[Source {number}: {source}, chunk {chunk}]\n"
        allowed = remaining - len(heading)
        if allowed <= 0:
            break
        section = heading + document.page_content[:allowed]
        sections.append(section)
        remaining -= len(section)
    return "\n\n---\n\n".join(sections)


def unique_documents(documents: list[Document]) -> list[Document]:
    """Keep order while removing duplicate chunks returned by retry searches."""
    seen: set[tuple[str, object]] = set()
    result = []
    for document in documents:
        key = (
            document.metadata.get("source_filename", ""),
            document.metadata.get("chunk_index", document.page_content),
        )
        if key not in seen:
            seen.add(key)
            result.append(document)
    return result
