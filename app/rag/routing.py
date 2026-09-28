from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import build_llm
from app.rag.routing_retriever import build_routing_retrievers


ROUTER_TEMPLATE = """Classify the user's question into exactly one category.

Categories:
- agent: questions about autonomous agents, planning, memory, tools, or agent loops.
- rag: questions about RAG, retrieval, embeddings, vector databases, HyDE, or step-back prompting.

Return only one word: agent or rag.

Question:
{question}
"""

ANSWER_TEMPLATE = """Answer the question using only the provided context.

Question:
{question}

Context:
---
{context}
---

Write a direct answer in 2 to 4 sentences.
If the context does not support the answer, say so clearly.
Do not use bullet points or tables.
"""

router_prompt = ChatPromptTemplate.from_template(ROUTER_TEMPLATE)
answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def route_question(question: str) -> str:
    llm = build_llm()

    chain = router_prompt | llm | StrOutputParser()
    route = chain.invoke({"question": question}).strip().lower()

    return route if route in {"agent", "rag"} else "rag"


def answer_with_routing(question: str, retrievers: dict) -> str:
    llm = build_llm()

    route = route_question(question)
    print(f"\nSelected route: {route}")

    docs = retrievers[route].invoke(question)
    context = format_docs(docs)

    answer_chain = answer_prompt | llm | StrOutputParser()

    return answer_chain.invoke(
        {
            "question": question,
            "context": context,
        }
    )


if __name__ == "__main__":
    retrievers = build_routing_retrievers()

    question = "What is HyDE and why is it used in RAG?"
    final_answer = answer_with_routing(question, retrievers)

    print("\nFinal answer:")
    print(final_answer)
