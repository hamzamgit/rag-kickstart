"""Cohere Adaptive RAG as the standard LangGraph adaptive-RAG workflow.

The graph intentionally follows the reference notebook's shape:

START -> route -> (retrieve -> grade_documents | web_search | llm_fallback)
                     -> generate -> grade_generation -> (END | retry | web_search)

Project-specific prompting, Cohere calls, retrieval, and Tavily conversion stay
in ``adaptive_rag.py``. This module only owns graph state and control flow.
"""

import argparse

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith import traceable

from app.agentic_rag.AdaptiveRAG.adaptive_rag import (
    answers_question,
    build_cohere_client,
    build_tavily_client,
    generate_answer,
    grade_document,
    is_grounded,
    search_web,
    select_route,
)
from app.agentic_rag.common import unique_documents
from app.rag.retriever import build_retriever


class AdaptiveRagGraphState(TypedDict, total=False):
    """Values carried between the adaptive-RAG nodes."""

    question: str
    client: object
    retriever: object
    tavily_client: object
    documents: list
    evidence: list
    attempts: list
    answer: str
    grounded: bool | None
    answers_question: bool | None
    retry_count: int
    max_retries: int
    failure_reason: str


def route_question(state: dict) -> str:
    """Choose the vector store or web, using the project's corpus description."""
    route = select_route(state["client"], state["question"])
    if route == "database-search":
        return "vectorstore"
    if route == "web-search":
        return "web_search"
    return "llm_fallback"


def retrieve(state: dict) -> dict:
    """Retrieve local documents for the original question."""
    return {"documents": state["retriever"].invoke(state["question"])}


def grade_documents(state: dict) -> dict:
    """Keep only local documents relevant to the question."""
    documents = state["documents"]
    document_grades = [
        {
            "document": document,
            "relevance": grade_document(state["client"], state["question"], document),
        }
        for document in documents
    ]
    relevant_documents = unique_documents(
        [grade["document"] for grade in document_grades if grade["relevance"] == "relevant"]
    )
    attempt = {
        "query": state["question"],
        "source": "vectorstore",
        "documents": documents,
        "document_grades": document_grades,
        "relevant_documents": relevant_documents,
    }
    return {
        "documents": relevant_documents,
        "evidence": relevant_documents,
        "attempts": [*state["attempts"], attempt],
    }


def decide_to_generate(state: dict) -> str:
    """Use the web when the vector store produced no relevant evidence."""
    return "generate" if state["documents"] else "web_search"


def web_search(state: dict) -> dict:
    """Search Tavily and put web results into the same document state field."""
    documents = search_web(
        state["tavily_client"] or build_tavily_client(), state["question"]
    )
    return {
        "documents": documents,
        "evidence": documents,
        "attempts": [
            *state["attempts"],
            {"query": state["question"], "source": "web-search", "documents": documents},
        ],
    }


def generate(state: dict) -> dict:
    """Generate from the current (local or web) documents."""
    documents = state["documents"]
    if not documents:
        return {
            "answer": "I could not find evidence for that question.",
            "grounded": False,
            "answers_question": False,
            "failure_reason": "No sources were available for generation.",
        }
    return {"answer": generate_answer(state["client"], state["question"], documents)}


def llm_fallback(_state: dict) -> dict:
    """End safely if a future router produces an unsupported route."""
    return {
        "answer": "I could not choose a reliable source for that question.",
        "grounded": None,
        "answers_question": None,
        "failure_reason": "The router returned an unsupported route.",
    }


def grade_generation(state: dict) -> dict:
    """Check that a generated answer is grounded and answers the question."""
    if not state["documents"]:
        return {
            "grounded": False,
            "answers_question": False,
            "retry_count": state["retry_count"] + 1,
            "failure_reason": "No sources were available for generation.",
        }

    grounded = is_grounded(state["client"], state["answer"], state["documents"])
    useful = (
        answers_question(state["client"], state["question"], state["answer"])
        if grounded
        else False
    )
    failed = not (grounded and useful)
    return {
        "grounded": grounded,
        "answers_question": useful,
        "retry_count": state["retry_count"] + int(failed),
        "failure_reason": (
            ""
            if not failed
            else "The generated answer was not grounded in the evidence."
            if not grounded
            else "The grounded answer did not sufficiently answer the question."
        ),
    }


def grade_generation_decision(state: dict) -> str:
    """Mirror the reference graph, with a bounded retry guard for production."""
    if state["grounded"] and state["answers_question"]:
        return "useful"
    if state["retry_count"] >= state["max_retries"]:
        return "stop"
    return "not_supported" if not state["grounded"] else "not_useful"


def stop(_state: dict) -> dict:
    """Produce a consistent terminal result after bounded retries."""
    return {
        "answer": "I could not produce a relevant, grounded, and useful answer.",
        "grounded": False,
        "answers_question": False,
    }


def build_adaptive_rag_graph():
    """Compile the reference adaptive-RAG topology using project dependencies."""
    workflow = StateGraph(AdaptiveRagGraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    workflow.add_node("grade_generation", grade_generation)
    workflow.add_node("llm_fallback", llm_fallback)
    workflow.add_node("stop", stop)

    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search",
            "llm_fallback": "llm_fallback",
        },
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "web_search": "web_search"},
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", "grade_generation")
    workflow.add_conditional_edges(
        "grade_generation",
        grade_generation_decision,
        {
            "useful": END,
            "not_supported": "generate",
            "not_useful": "web_search",
            "stop": "stop",
        },
    )
    workflow.add_edge("llm_fallback", END)
    workflow.add_edge("stop", END)
    return workflow.compile()


@traceable(name="cohere-langgraph-adaptive-rag", run_type="chain")
def answer_with_langgraph_adaptive_rag(
    question: str,
    retriever=None,
    client=None,
    tavily_client=None,
    *,
    max_retries: int = 2,
) -> dict:
    """Run one complete adaptive-RAG graph execution."""
    if not question or not question.strip():
        raise ValueError("question must not be empty")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    question = question.strip()
    initial_state = {
        "question": question,
        "client": client or build_cohere_client(),
        "retriever": retriever or build_retriever(k=4),
        "tavily_client": tavily_client,
        "documents": [],
        "evidence": [],
        "attempts": [],
        "answer": "",
        "grounded": None,
        "answers_question": None,
        "retry_count": 0,
        "max_retries": max_retries,
        "failure_reason": "",
    }
    return build_adaptive_rag_graph().invoke(initial_state)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question through Cohere LangGraph Adaptive RAG."
    )
    parser.add_argument("question", help="Question to answer.")
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    result = answer_with_langgraph_adaptive_rag(args.question, max_retries=args.max_retries)
    print(f"Grounded: {result['grounded']}; answers question: {result['answers_question']}")
    print(f"\nAnswer:\n{result['answer']}")


if __name__ == "__main__":
    main()
