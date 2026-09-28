"""Semantic routing using only vectors that are already in PostgreSQL.

Unlike the old version, this module does not create FAISS vectors for example
questions. It searches the stored document chunks and uses the nearest chunk's
topic as the route. Therefore every vector search uses pgvector.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import build_llm
from app.rag.retriever import build_retriever
from app.rag.structured_routing import build_local_retrievers, format_docs


ANSWER_TEMPLATE = """Answer the question using only the retrieved context.

Question:
{question}

Semantically selected local topic:
{topic}

Retrieved context:
---
{context}
---

Give a direct answer in 2 to 4 sentences. If the context does not contain
enough information, say so clearly. Do not use bullets, tables, or code blocks.
"""

answer_prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)


def build_semantic_router():
    """Build the broad pgvector search used to select a topic.

    This replaces the old FAISS router. Asking for more than one candidate lets
    us compare the best distance seen for each topic before choosing a route.
    """
    return build_retriever(k=20)


def semantic_route(question: str, router) -> str:
    """Choose the topic whose stored chunk is closest to the question.

    Cosine distance is smaller when embeddings are more similar. The shared
    retriever includes that distance in each returned document's metadata.
    """
    closest_documents = router.invoke(question)

    if not closest_documents:
        raise ValueError("No stored chunks are available for semantic routing.")

    closest_distance_by_topic = {}

    for document in closest_documents:
        topic = document.metadata["topic"]
        distance = document.metadata["distance"]
        previous_distance = closest_distance_by_topic.get(topic)

        if previous_distance is None or distance < previous_distance:
            closest_distance_by_topic[topic] = distance

    selected_topic = min(closest_distance_by_topic, key=closest_distance_by_topic.get)

    print("\nClosest stored chunk:")
    print(closest_documents[0].page_content[:180].replace("\n", " "))
    print("Selected local topic:", selected_topic)

    return selected_topic


def answer_with_semantic_routing(question: str, router, retrievers: dict) -> str:
    """Semantically route without an LLM, retrieve locally, then answer."""
    llm = build_llm()

    # 1. Select a topic from the nearest chunks stored in PostgreSQL.
    selected_topic = semantic_route(question, router)

    # 2. Search only within that topic for the final answer context.
    selected_retriever = retrievers[selected_topic]
    docs = selected_retriever.invoke(question)
    context = format_docs(docs)

    print("\nRetrieved chunks:")
    for doc in docs:
        print("-", doc.page_content[:180].replace("\n", " "))

    # 3. Use the LLM only to write an answer grounded in those chunks.
    answer_chain = answer_prompt | llm | StrOutputParser()

    return answer_chain.invoke(
        {
            "question": question,
            "topic": selected_topic,
            "context": context,
        }
    )


if __name__ == "__main__":
    semantic_router = build_semantic_router()
    local_retrievers = build_local_retrievers()
    question = "What is the purpose of self-reflection in Self-RAG?"
    final_answer = answer_with_semantic_routing(
        question=question,
        router=semantic_router,
        retrievers=local_retrievers,
    )

    print("\nFinal answer:")
    print(final_answer)
