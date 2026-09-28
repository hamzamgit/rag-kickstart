"""Query construction with local pgvector retrieval and database metadata."""

from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.llm import build_llm
from app.rag.retriever import build_retriever


class SearchRequest(BaseModel):
    """A semantic query plus filters that exist in the local documents table."""

    search_query: str = Field(
        ...,
        description="A short semantic search query for the user's question.",
    )
    topic: Literal["rag", "hyde", "agents", "self_rag"] | None = Field(
        default=None,
        description="Use only if the user explicitly asks about one local topic.",
    )
    max_published_year: int | None = Field(
        default=None,
        description="Use only when the user explicitly asks for work before a year.",
    )


QUERY_CONSTRUCTION_TEMPLATE = """Convert the question into a semantic search
query and metadata filters for the local research-document database.

You may use only these local metadata fields:
- topic: rag, hyde, agents, self_rag
- max_published_year: an integer year

Rules:
- Never invent a metadata constraint.
- Set topic only when the user explicitly identifies one topic.
- Set max_published_year only when the user explicitly says "before YEAR".
- Do not answer the question.

User question:
{question}
"""

ANSWER_TEMPLATE = """Answer the user's question using only the retrieved context.

User question:
{question}

Search query:
{search_query}

Applied database filters:
- topic: {topic}
- maximum publication year: {max_published_year}

Retrieved context:
---
{context}
---

Give a direct answer in 2 to 4 sentences. If no matching documents were
retrieved, state that clearly. Do not use tables, bullets, or code blocks.
"""

query_construction_prompt = ChatPromptTemplate.from_template(
    QUERY_CONSTRUCTION_TEMPLATE
)
answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)


def format_docs(docs) -> str:
    """Turn retrieved documents into one context string for the answer prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_database_retriever():
    """Build the shared local retriever used by this example.

    Per-question filters are passed later to ``invoke``. This mirrors the old
    example's one combined vector store while avoiding any in-memory index.
    """
    return build_retriever(k=3)


def construct_search_request(question: str) -> SearchRequest:
    """Use structured LLM output to create a safe, validated search request."""
    llm = build_llm()
    structured_llm = llm.with_structured_output(SearchRequest)
    query_construction_chain = query_construction_prompt | structured_llm

    return query_construction_chain.invoke({"question": question})


def retrieve_documents(retriever, request: SearchRequest):
    """Search pgvector and apply supported filters inside the SQL query.

    ``topics`` becomes ``WHERE d.topic = ANY(...)`` and the year becomes
    ``WHERE d.published_year <= ...``. The filtering therefore happens before
    PostgreSQL selects the nearest chunks, not afterwards in Python.
    """
    topic_filter = (request.topic,) if request.topic else None

    return retriever.invoke(
        request.search_query,
        topics=topic_filter,
        max_published_year=request.max_published_year,
    )


def answer_with_query_construction(question: str, retriever) -> str:
    """Construct a request, retrieve matching local chunks, and answer."""
    llm = build_llm()

    # 1. Convert natural language into a semantic query and SQL-safe filters.
    request = construct_search_request(question)

    print("\nConstructed search query:", request.search_query)
    print("Topic filter:", request.topic)
    print("Maximum publication year:", request.max_published_year)

    # 2. Query the local pgvector database with those filters.
    docs = retrieve_documents(retriever, request)
    context = format_docs(docs)

    print("\nRetrieved chunks:")
    for document in docs:
        print("-", document.page_content[:180].replace("\n", " "))

    if not context:
        return (
            "No documents matched the requested metadata constraints. "
            "Ingest a relevant document before answering this query."
        )

    # 3. Generate an answer from the database context only.
    answer_chain = answer_prompt | llm | StrOutputParser()

    return answer_chain.invoke(
        {
            "question": question,
            "search_query": request.search_query,
            "topic": request.topic or "No filter",
            "max_published_year": request.max_published_year or "No filter",
            "context": context,
        }
    )


if __name__ == "__main__":
    database_retriever = build_database_retriever()
    question = "How did retrieval-augmented generation work before 2021?"
    final_answer = answer_with_query_construction(question, database_retriever)

    print("\nFinal answer:")
    print(final_answer)
