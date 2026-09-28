"""ColBERTv2 indexing and answer generation for the local document chunks."""

import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.indexing.colbert.colbert_retriever import (
    build_colbert_index,
    build_colbert_retriever,
)
from app.llm import build_llm


ANSWER_TEMPLATE = """Answer using only the original ColBERT-retrieved chunks.

Question:
{question}

Retrieved context:
---
{context}
---

Give a direct answer in 3 to 5 sentences. If the context does not contain the
answer, say so clearly. Do not use bullets, tables, or code blocks.
"""

answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)


def format_docs(docs) -> str:
    """Keep the original chunks and ColBERT ranks visible to the final LLM."""
    return "\n\n".join(
        f"[ColBERT rank {document.metadata['colbert_rank']}]\n"
        f"{document.page_content}"
        for document in docs
    )


def answer_with_colbert(question: str, retriever) -> str:
    """Use MaxSim for retrieval and call an LLM only for the final answer."""
    # 1. ColBERT creates query token embeddings and runs late interaction.
    docs = retriever.invoke(question)
    context = format_docs(docs)

    print("\nRetrieved ColBERT chunks:")
    for document in docs:
        print(
            f"- rank {document.metadata['colbert_rank']}; "
            f"score {document.metadata['colbert_score']:.3f}: "
            f"{document.page_content[:180].replace(chr(10), ' ')}"
        )

    if not context:
        return "No ColBERT chunks are indexed yet. Build the ColBERT index first."

    # 2. No LLM was used in indexing or retrieval; it is used only here.
    llm = build_llm(max_tokens=600, reasoning_effort="low")
    answer_chain = answer_prompt | llm | StrOutputParser()
    return answer_chain.invoke({"question": question, "context": context})


def main():
    parser = argparse.ArgumentParser(description="Build and query a ColBERTv2 index.")
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Export database chunks and rebuild the on-disk ColBERT index.",
    )
    args = parser.parse_args()

    if args.build_index:
        build_colbert_index()
        return

    colbert_retriever = build_colbert_retriever(k=5)
    question = "How does Self-RAG decide whether retrieval is useful?"
    final_answer = answer_with_colbert(question, colbert_retriever)

    print("\nFinal answer:")
    print(final_answer)


if __name__ == "__main__":
    main()
