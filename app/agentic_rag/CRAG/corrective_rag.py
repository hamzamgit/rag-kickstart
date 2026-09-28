"""Corrective RAG (CRAG): assess retrieval, correct it, then answer."""

import argparse

from langsmith import traceable

from app.agentic_rag.common import format_documents, invoke_text, unique_documents
from app.llm import build_llm
from app.rag.retriever import build_retriever


RELEVANCE_TEMPLATE = """Decide whether this one retrieved source contains
useful evidence for answering the question. Reply with exactly one word:
relevant or irrelevant. A source is relevant only when it addresses the
question rather than merely sharing a broad topic.

Question: {question}

Source:
{context}
"""

REWRITE_TEMPLATE = """Rewrite this question as one short, precise search query
for a technical RAG knowledge base. Preserve the user's intent. Include key
terms that are likely to appear in source documents. Return only the query.

Question: {question}
"""

ANSWER_TEMPLATE = """Answer the question using only the supplied evidence.
If the evidence is insufficient, say what is missing instead of guessing.
Write a direct answer in 2 to 5 sentences, and cite supporting source numbers
in square brackets when making factual claims.

Question: {question}

Evidence:
{context}
"""

DEFAULT_QUESTION = "How does Self-RAG decide to retrieve?"


@traceable(name="retrieve-local-documents", run_type="retriever")
def retrieve_documents(retriever, query: str) -> list:
    """Retrieve local chunks in a named child span of the CRAG trace."""
    return retriever.invoke(query)


@traceable(name="grade-source-relevance", run_type="chain")
def grade_document(llm, question: str, document) -> str:
    """Grade one retrieved source, as required by the CRAG filter step."""
    response = invoke_text(
        llm,
        RELEVANCE_TEMPLATE,
        question=question,
        context=format_documents([document]),
    ).lower()
    return "relevant" if response == "relevant" else "irrelevant"


@traceable(name="rewrite-search-query", run_type="chain")
def rewrite_query(llm, question: str) -> str:
    """Create the single bounded corrective query."""
    return invoke_text(llm, REWRITE_TEMPLATE, question=question)


@traceable(name="generate-grounded-answer", run_type="chain")
def generate_answer(llm, question: str, evidence: list) -> str:
    """Generate an answer only from the sources that passed CRAG grading."""
    return invoke_text(
        llm,
        ANSWER_TEMPLATE,
        question=question,
        context=format_documents(evidence),
    )


@traceable(name="corrective-rag-run", run_type="chain")
def answer_with_crag(question: str, retriever=None, *, max_corrections: int = 1) -> dict:
    """Answer with one retrieval assessment and bounded corrective retries.

    CRAG is deliberately bounded: a bad search can trigger a query correction,
    but it cannot consume unbounded model calls.  External web search is left
    out because this local learning project uses its PostgreSQL corpus as the
    source of truth. When invoked from the CLI, this is the one LangSmith
    top-level trace; retrieval and Groq/LCEL activity is nested below it.
    """
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    if max_corrections < 0:
        raise ValueError("max_corrections must be non-negative")

    llm = build_llm(max_tokens=600)
    retriever = retriever or build_retriever(k=4)
    attempts = []
    query = question.strip()

    for correction_number in range(max_corrections + 1):
        documents = retrieve_documents(retriever, query)
        document_grades = [
            {"document": document, "relevance": grade_document(llm, question, document)}
            for document in documents
        ]
        relevant_documents = [
            grade["document"]
            for grade in document_grades
            if grade["relevance"] == "relevant"
        ]
        relevance = "relevant" if relevant_documents else "irrelevant"
        attempts.append(
            {
                "query": query,
                "documents": documents,
                "relevant_documents": relevant_documents,
                "document_grades": document_grades,
                "relevance": relevance,
            }
        )
        if relevance == "relevant":
            break
        if correction_number < max_corrections:
            query = rewrite_query(llm, question)

    evidence = unique_documents(
        [
            document
            for attempt in attempts
            for document in attempt["relevant_documents"]
        ]
    )
    if evidence:
        answer = generate_answer(llm, question, evidence)
    else:
        answer = "I could not find evidence for that question in the local corpus."

    route = "corrected-retrieval" if len(attempts) > 1 else "direct-retrieval"

    return {"answer": answer, "route": route, "attempts": attempts}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question through Corrective RAG.")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=f"Question to answer from the local corpus (default: {DEFAULT_QUESTION!r}).",
    )
    parser.add_argument("--max-corrections", type=int, default=1)
    args = parser.parse_args()
    result = answer_with_crag(args.question, max_corrections=args.max_corrections)
    print(f"Route: {result['route']}")
    for attempt in result["attempts"]:
        print(f"- {attempt['relevance']}: {attempt['query']}")
        for grade in attempt["document_grades"]:
            source = grade["document"].metadata.get("source_filename", "unknown source")
            print(f"  - {grade['relevance']}: {source}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
