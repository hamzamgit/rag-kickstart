from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.rag.decomposition import decompose
from app.llm import build_llm


TEMPLATE = """Here is the question you need to answer:

---
{question}
---

Here are any available background question-and-answer pairs:

---
{q_a_pairs}
---

Here is additional context relevant to the question:

---
{context}
---

Answer using only the supplied context and background answers.
If the answer is not supported by the context, say that clearly.
Keep the answer brief: 2 to 3 sentences maximum.
Do not use tables, bullet points, or code blocks.

Question: {question}
"""

SYNTHESIS_TEMPLATE = """You broke down the following question into sub-questions and answered each one.

Original question:
{question}

Sub-question answers:
---
{q_a_pairs}
---

Using only the information above, write one direct and complete answer to
the original question. Do not add unsupported claims.
Keep it to 4 to 5 sentences maximum. Plain prose only.
"""

decomposition_prompt = ChatPromptTemplate.from_template(TEMPLATE)
synthesis_prompt = ChatPromptTemplate.from_template(SYNTHESIS_TEMPLATE)


def format_qa_pair(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def answer_recursively(main_question: str, retriever) -> str:
    llm = build_llm()
    questions = decompose(main_question)

    q_a_pairs = []

    for question in questions:
        retrieved_docs = retriever.invoke(question)
        context = format_docs(retrieved_docs)

        print(f"\nRetrieved context for: {question}")
        for doc in retrieved_docs:
            print("-", doc.page_content[:200].replace("\n", " "))

        rag_chain = (
            {
                "context": lambda _: context,
                "question": itemgetter("question"),
                "q_a_pairs": itemgetter("q_a_pairs"),
            }
            | decomposition_prompt
            | llm
            | StrOutputParser()
        )

        previous_answers = "\n\n---\n\n".join(q_a_pairs) or "No previous answers yet."

        answer = rag_chain.invoke(
            {
                "question": question,
                "q_a_pairs": previous_answers,
            }
        )

        q_a_pairs.append(format_qa_pair(question, answer))

    synthesis_chain = synthesis_prompt | llm | StrOutputParser()

    return synthesis_chain.invoke(
        {
            "question": main_question,
            "q_a_pairs": "\n\n---\n\n".join(q_a_pairs),
        }
    )


if __name__ == "__main__":
    from app.rag.retriever import build_retriever

    retriever = build_retriever()

    question = "How does Self-RAG decide whether retrieval is useful?"

    final_answer = answer_recursively(question, retriever)

    print("\nFinal answer:")
    print(final_answer)
