"""Structured routing over the local PostgreSQL and pgvector document store.

This is the local equivalent of the previous online structured-routing example.
The important difference is only the retrieval source: it now reads the chunks
that ``app.ingestion.ingest_documents_to_postgres`` has already stored.
"""

from typing import Literal

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.llm import build_llm
from app.rag.retriever import build_retriever


DocumentTopic = Literal["rag", "hyde", "agents", "self_rag"]


class RouteQuery(BaseModel):
    """The validated routing decision returned by the LLM."""

    topic: DocumentTopic = Field(
        ...,
        description="The local document topic most relevant to the question.",
    )


ROUTER_SYSTEM_PROMPT = """Route the user's question to one local document topic.

- rag: foundational retrieval-augmented generation research.
- hyde: hypothetical document embeddings for zero-shot retrieval.
- agents: ReAct, reasoning, actions, and tool use by language-model agents.
- self_rag: retrieval, generation, and self-reflection or critique.
"""

ANSWER_TEMPLATE = """Answer the question using only the retrieved context.

Question:
{question}

Selected local topic:
{topic}

Retrieved context:
---
{context}
---

Give a direct answer in 2 to 4 sentences. If the context does not contain the
answer, say so clearly. Do not use tables, bullets, or code blocks.
"""

router_prompt = ChatPromptTemplate.from_messages(
    [("system", ROUTER_SYSTEM_PROMPT), ("human", "{question}")]
)
answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)


def format_docs(docs) -> str:
    """Join retrieved LangChain documents into context for the answer prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_local_retrievers() -> dict:
    """Build one pgvector retriever for each topic in the local database.

    These objects do not make a database request yet. A request happens later
    when ``retriever.invoke(question)`` embeds the question and runs SQL.
    """
    topics = ("rag", "hyde", "agents", "self_rag")
    retrievers = {}

    for topic in topics:
        retrievers[topic] = build_retriever(k=3, topics=(topic,))

    return retrievers


def route_question(llm, question: str) -> RouteQuery:
    """Ask the LLM for a validated topic instead of accepting free-form text."""
    structured_llm = llm.with_structured_output(RouteQuery)
    router_chain = router_prompt | structured_llm

    return router_chain.invoke({"question": question})


def answer_with_structured_routing(question: str, retrievers: dict) -> str:
    """Route, retrieve from the selected local topic, then answer from context."""
    llm = build_llm()

    # 1. Route the question. RouteQuery guarantees that topic is one of the
    #    four valid topic names defined above.
    route_result = route_question(llm, question)
    topic = route_result.topic

    print(f"\nSelected local topic: {topic}")

    # 2. Only query chunks belonging to the selected topic in PostgreSQL.
    selected_retriever = retrievers[topic]
    docs = selected_retriever.invoke(question)
    context = format_docs(docs)

    print("\nRetrieved chunks:")
    for doc in docs:
        print("-", doc.page_content[:180].replace("\n", " "))

    # 3. Give the LLM only the retrieved database context for the final answer.
    answer_chain = answer_prompt | llm | StrOutputParser()

    return answer_chain.invoke(
        {"question": question, "topic": topic, "context": context}
    )


if __name__ == "__main__":
    local_retrievers = build_local_retrievers()
    question = "How does ReAct combine reasoning with actions?"
    final_answer = answer_with_structured_routing(question, local_retrievers)

    print("\nFinal answer:")
    print(final_answer)
