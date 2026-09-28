from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import build_llm
from app.rag.retriever import build_retriever


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


HYDE_TEMPLATE = """Write a short hypothetical reference document that could
answer the user's question.

This is only for document retrieval, not the final answer.
Do not mention that it is hypothetical.
Write 1 or 2 factual-looking paragraphs.

Question:
{question}
"""

ANSWER_TEMPLATE = """Answer the user's question using only the retrieved context.

Question:
{question}

Retrieved context:
---
{context}
---

Give a direct answer in 3 to 5 sentences. If the retrieved context does not
contain enough information, say so clearly. Do not use bullets or tables.
"""

hyde_prompt = ChatPromptTemplate.from_template(HYDE_TEMPLATE)
answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)


def answer_with_hyde(question: str, retriever) -> str:
    llm = build_llm()

    # 1. Generate a hypothetical answer/document.
    hyde_chain = hyde_prompt | llm | StrOutputParser()
    hypothetical_document = hyde_chain.invoke({"question": question})

    print("\nHyDE hypothetical document:")
    print(hypothetical_document)

    # 2. Search with the generated document, not only the original question.
    retrieved_docs = retriever.invoke(hypothetical_document)
    context = format_docs(retrieved_docs)

    print("\nRetrieved context:")
    for doc in retrieved_docs:
        print("-", doc.page_content)

    # 3. Produce the real answer only from your actual retrieved data.
    answer_chain = answer_prompt | llm | StrOutputParser()

    return answer_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )


if __name__ == "__main__":
    retriever = build_retriever()

    question = "How does an autonomous AI agent use planning, memory, and tools?"
    final_answer = answer_with_hyde(question, retriever)

    print("\nFinal answer:")
    print(final_answer)
