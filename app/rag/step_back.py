from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from app.llm import build_llm
from app.rag.retriever import build_retriever


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


examples = [
    {
        "input": "Could members of the police perform lawful arrests?",
        "output": "What are the legal powers and duties of police officers?",
    },
    {
        "input": "Jan Šindel was born in what country?",
        "output": "What is Jan Šindel's personal history?",
    },
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

step_back_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You transform a specific question into a broader question.
The broader question should help retrieve background knowledge needed to answer
the original question. Return only the rewritten question.""",
        ),
        few_shot_prompt,
        ("human", "{question}"),
    ]
)

answer_prompt = ChatPromptTemplate.from_template(
    """Answer the user's question using only the provided context.

Original question:
{question}

Context retrieved using the original question:
---
{normal_context}
---

Context retrieved using the broader step-back question:
---
{step_back_context}
---

Give a direct answer in 3 to 5 sentences. If the context does not support an
answer, say so clearly. Do not use bullets or tables.
"""
)


def answer_with_step_back(question: str, retriever) -> str:
    llm = build_llm()

    step_back_chain = step_back_prompt | llm | StrOutputParser()
    step_back_question = step_back_chain.invoke({"question": question})

    normal_docs = retriever.invoke(question)
    step_back_docs = retriever.invoke(step_back_question)

    print(f"\nOriginal question: {question}")
    print(f"Step-back question: {step_back_question}")

    answer_chain = answer_prompt | llm | StrOutputParser()

    return answer_chain.invoke(
        {
            "question": question,
            "normal_context": format_docs(normal_docs),
            "step_back_context": format_docs(step_back_docs),
        }
    )


if __name__ == "__main__":
    retriever = build_retriever()

    question = "How does Self-RAG decide whether retrieval is useful?"
    final_answer = answer_with_step_back(question, retriever)

    print("\nFinal answer:")
    print(final_answer)
