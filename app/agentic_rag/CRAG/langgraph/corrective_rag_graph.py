"""A LangGraph implementation of Corrective RAG (CRAG).

The graph uses ordinary dictionaries for its state so every value is easy to
inspect in a debugger or LangSmith trace.  It makes one relevance LLM call per
retrieved document.  It retries with a rewritten query only when no document
passes that relevance filter.
"""

import argparse
from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app.agentic_rag.common import unique_documents
from app.agentic_rag.CRAG.corrective_rag import (
    DEFAULT_QUESTION,
    generate_answer,
    grade_document,
    retrieve_documents,
    rewrite_query,
)
from app.llm import build_llm
from app.rag.retriever import build_retriever


class CragGraphState(TypedDict, total=False):
    """LangGraph's key map; values passed between nodes remain plain dictionaries."""

    question: str
    query: str
    retriever: object
    llm: object
    max_corrections: int
    correction_count: int
    documents: list
    attempts: list
    evidence: list
    answer: str
    route: str


def retrieve_node(state: dict) -> dict:
    """Retrieve chunks for the current query."""
    documents = retrieve_documents(state["retriever"], state["query"])
    return {"documents": documents}


def grade_sources_node(state: dict) -> dict:
    """Grade every source and retain only the sources that pass CRAG filtering."""
    document_grades = [
        {
            "document": document,
            "relevance": grade_document(state["llm"], state["question"], document),
        }
        for document in state["documents"]
    ]
    relevant_documents = [
        grade["document"]
        for grade in document_grades
        if grade["relevance"] == "relevant"
    ]
    attempt = {
        "query": state["query"],
        "documents": state["documents"],
        "document_grades": document_grades,
        "relevant_documents": relevant_documents,
        "relevance": "relevant" if relevant_documents else "irrelevant",
    }
    return {
        "attempts": [*state["attempts"], attempt],
        "evidence": unique_documents([*state["evidence"], *relevant_documents]),
    }


def choose_next_step(state: dict) -> str:
    """Continue only when every retrieved source was irrelevant."""
    if state["evidence"]:
        return "generate_answer"
    if state["correction_count"] < state["max_corrections"]:
        return "rewrite_query"
    return "no_evidence"


def rewrite_query_node(state: dict) -> dict:
    """Correct the query and return it to the retrieval node."""
    return {
        "query": rewrite_query(state["llm"], state["question"]),
        "correction_count": state["correction_count"] + 1,
    }


def generate_answer_node(state: dict) -> dict:
    """Generate from only the documents that passed the relevance filter."""
    route = "corrected-retrieval" if state["correction_count"] else "direct-retrieval"
    return {
        "answer": generate_answer(state["llm"], state["question"], state["evidence"]),
        "route": route,
    }


def no_evidence_node(state: dict) -> dict:
    """Stop cleanly after the configured number of unsuccessful corrections."""
    route = "corrected-retrieval" if state["correction_count"] else "direct-retrieval"
    return {
        "answer": "I could not find evidence for that question in the local corpus.",
        "route": route,
    }


def build_corrective_rag_graph():
    """Compile the CRAG graph with explicit correction-loop routing."""
    graph = StateGraph(CragGraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_sources", grade_sources_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("no_evidence", no_evidence_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_sources")
    graph.add_conditional_edges(
        "grade_sources",
        choose_next_step,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
            "no_evidence": "no_evidence",
        },
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", END)
    graph.add_edge("no_evidence", END)
    return graph.compile()


@traceable(name="langgraph-corrective-rag-run", run_type="chain")
def answer_with_langgraph_crag(
    question: str,
    retriever=None,
    *,
    max_corrections: int = 1,
) -> dict:
    """Run one complete LangGraph CRAG workflow and return its final state."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    if max_corrections < 0:
        raise ValueError("max_corrections must be non-negative")

    initial_state = {
        "question": question.strip(),
        "query": question.strip(),
        "retriever": retriever or build_retriever(k=4),
        "llm": build_llm(max_tokens=600),
        "max_corrections": max_corrections,
        "correction_count": 0,
        "documents": [],
        "attempts": [],
        "evidence": [],
        "answer": "",
        "route": "",
    }
    return build_corrective_rag_graph().invoke(initial_state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question through LangGraph CRAG.")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help=f"Question to answer from the local corpus (default: {DEFAULT_QUESTION!r}).",
    )
    parser.add_argument("--max-corrections", type=int, default=1)
    args = parser.parse_args()

    result = answer_with_langgraph_crag(
        args.question, max_corrections=args.max_corrections
    )
    print(f"Route: {result['route']}")
    for attempt in result["attempts"]:
        print(f"- {attempt['relevance']}: {attempt['query']}")
        for grade in attempt["document_grades"]:
            source = grade["document"].metadata.get("source_filename", "unknown source")
            print(f"  - {grade['relevance']}: {source}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
