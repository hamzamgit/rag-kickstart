"""Real multi-vector retrieval: many vectors for every stored chunk.

Normal chunk retrieval stores one embedding for each chunk:

    chunk text -> one vector

This example stores eight embeddings for each *same* chunk:

    original chunk text -> vector 1
    chunk summary       -> vector 2
    key concepts        -> vector 3
    hypothetical query  -> vectors 4 through 8

All eight rows point to the same ``document_chunks.id`` parent. At search time,
one user-query vector is compared with every representation vector. The best
matching representation determines the score for its parent chunk.
"""

import argparse
import re
import time

from groq import APIStatusError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import build_llm
from app.indexing.multivector.multi_vector_retriever import build_multi_vector_retriever


REPRESENTATION_TEMPLATE = """Create additional retrieval representations for
one chunk from a local research document.

Document title: {title}
Document topic: {topic}

Chunk text:
---
{chunk}
---

Return exactly this plain-text format, with no Markdown headings or bullets:

SUMMARY: one factual summary
KEY_CONCEPTS: concise searchable concepts
QUESTIONS:
1. first question
2. second question
3. third question
4. fourth question
5. fifth question

Do not add information that is not in the chunk. Do not answer the questions.
"""

ANSWER_TEMPLATE = """Answer the user's question using only the retrieved context.

Question:
{question}

Retrieved context:
---
{context}
---

Give a direct answer in 3 to 5 sentences. If the context does not contain
enough information, say so clearly. Do not use bullets, tables, or code blocks.
"""

representation_prompt = ChatPromptTemplate.from_template(REPRESENTATION_TEMPLATE)
answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

# A 1,000-character chunk plus this short response stays well below the
# per-request budget. The pause keeps repeated indexing calls below Groq's
# 8,000 tokens-per-minute quota on the on-demand tier.
MULTI_VECTOR_MAX_TOKENS = 400
MULTI_VECTOR_REQUEST_INTERVAL_SECONDS = 8
GROQ_RATE_LIMIT_RETRY_SECONDS = 60


def format_docs(docs) -> str:
    """Join full documents or chunks after multi-vector search for the answer."""
    return "\n\n".join(doc.page_content for doc in docs)


def parse_chunk_representations(llm_output: str) -> tuple[str, str, list[str]]:
    """Parse the explicit text format without requiring an LLM tool call.

    Groq models sometimes generate a correct answer but decline a forced tool
    call. Plain text plus local validation is more reliable for this batch job.
    """
    summary_match = re.search(
        r"SUMMARY:\s*(.+?)\s*KEY_CONCEPTS:",
        llm_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    concepts_match = re.search(
        r"KEY_CONCEPTS:\s*(.+?)\s*QUESTIONS:",
        llm_output,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not summary_match or not concepts_match:
        raise ValueError(
            "The LLM did not return the required SUMMARY, KEY_CONCEPTS, and "
            "QUESTIONS format. Retry this missing chunk."
        )

    questions_match = re.search(
        r"QUESTIONS:\s*(.*)$",
        llm_output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not questions_match:
        raise ValueError("The LLM output is missing the QUESTIONS section.")

    questions_section = questions_match.group(1)
    questions = []

    for line in questions_section.splitlines():
        question_match = re.match(r"\s*\d+[.)]\s*(.+)", line)
        if question_match:
            questions.append(question_match.group(1).strip())

    if len(questions) != 5:
        raise ValueError(
            "The LLM must return exactly five hypothetical questions; "
            f"received {len(questions)}. Retry this missing chunk."
        )

    return (
        summary_match.group(1).strip(),
        concepts_match.group(1).strip(),
        questions,
    )


def create_representations(llm, chunk: dict) -> list[tuple[str, int, str]]:
    """Create eight textual representations for one existing database chunk.

    The first representation is the original chunk itself. The next seven are
    alternate views of the same chunk, not chunks from different source text.
    """
    representation_chain = representation_prompt | llm | StrOutputParser()
    llm_output = representation_chain.invoke(
        {
            "title": chunk["title"],
            "topic": chunk["topic"],
            "chunk": chunk["content"],
        }
    )
    summary, key_concepts, hypothetical_questions = parse_chunk_representations(
        llm_output
    )

    representations = [
        ("original", 0, chunk["content"]),
        ("summary", 0, summary),
        ("key_concepts", 0, key_concepts),
    ]

    for question_index, hypothetical_question in enumerate(hypothetical_questions):
        representations.append(
            ("hypothetical_question", question_index, hypothetical_question)
        )

    return representations


def index_missing_chunk_representations(*, limit: int | None = None) -> int:
    """Generate and store multi-vectors for chunks not indexed by this example.

    Normal document ingestion and document backfill call this function
    automatically after their document/chunk work completes. You can still run
    it directly when recovering an interrupted indexing job.
    """
    # The representation format is short. Limiting output prevents Groq from
    # reserving its default 8,192-token completion budget for every chunk.
    llm = build_llm(
        max_tokens=MULTI_VECTOR_MAX_TOKENS,
        reasoning_effort="low",
    )
    multi_vector_retriever = build_multi_vector_retriever()
    multi_vector_retriever.ensure_schema()
    chunks = multi_vector_retriever.chunks_without_representations(limit=limit)

    if not chunks:
        print("Every stored chunk already has multi-vector representations.")
        return 0

    for number, chunk in enumerate(chunks, start=1):
        print(
            f"Creating 8 representations for chunk {number}/{len(chunks)}: "
            f"{chunk['source_filename']} (chunk {chunk['chunk_id']})"
        )

        # One call can be rejected if an earlier run has already consumed the
        # rolling TPM budget. Wait for that window to reset, then retry the
        # same missing chunk. Nothing is written before representations exist.
        while True:
            try:
                representations = create_representations(llm, chunk)
                break
            except APIStatusError as error:
                if error.status_code != 413:
                    raise

                print(
                    "Groq TPM limit reached. Waiting "
                    f"{GROQ_RATE_LIMIT_RETRY_SECONDS} seconds before retrying "
                    "this same chunk."
                )
                time.sleep(GROQ_RATE_LIMIT_RETRY_SECONDS)

        multi_vector_retriever.replace_representations(
            chunk_id=chunk["chunk_id"],
            representations=representations,
        )

        if number < len(chunks):
            time.sleep(MULTI_VECTOR_REQUEST_INTERVAL_SECONDS)

    print(f"Stored {len(chunks) * 8} representation vectors for {len(chunks)} chunks.")
    return len(chunks)


def answer_with_multi_vector_retrieval(question: str, retriever) -> str:
    """Search one query vector against all representation vectors in pgvector."""
    llm = build_llm(max_tokens=600, reasoning_effort="low")

    # 1. The user question becomes ONE query vector.
    # 2. PostgreSQL compares it with the eight stored vectors for every chunk.
    # 3. The best matching representation selects each parent chunk's score.
    docs = retriever.invoke(question)
    context = format_docs(docs)

    print("\nRetrieved context:")
    for document in docs:
        matched_type = document.metadata["matched_representation_type"]
        matched_index = document.metadata["matched_representation_index"]
        retrieval_scope = document.metadata["retrieval_scope"]
        print(
            f"- {retrieval_scope}; matched {matched_type}[{matched_index}]: "
            f"{document.page_content[:160].replace(chr(10), ' ')}"
        )

    if not context:
        return (
            "No multi-vector representations are available yet. Run the indexing "
            "command first."
        )

    # 4. The LLM receives full documents when short, otherwise original chunks.
    answer_chain = answer_prompt | llm | StrOutputParser()
    return answer_chain.invoke({"question": question, "context": context})


def main():
    parser = argparse.ArgumentParser(description="Learn per-chunk multi-vector RAG.")
    parser.add_argument(
        "--index-missing",
        action="store_true",
        help="Recover by generating vectors only for still-unindexed chunks.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit how many chunks are indexed; useful for a low-cost first run.",
    )
    args = parser.parse_args()

    if args.index_missing:
        index_missing_chunk_representations(limit=args.limit)
        return

    # The current PDFs are all at or below the default 40-page limit, so this
    # returns their full stored text. Lower the limit for larger collections.
    multi_vector_retriever = build_multi_vector_retriever(k=3, full_document_page_limit=1)
    multi_vector_retriever.ensure_schema()

    question = "How does Self-RAG decide whether retrieval is useful?"
    final_answer = answer_with_multi_vector_retrieval(
        question=question,
        retriever=multi_vector_retriever,
    )

    print("\nFinal answer:")
    print(final_answer)


if __name__ == "__main__":
    main()
