from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.llm import build_llm

DECOMPOSITION_TEMPLATE = """
Break the user's question into at most 3 smaller factual sub-questions.

Rules:
- Each question must help answer the original question.
- Do not repeat the original question.
- Return only numbered questions.
- If decomposition is unnecessary, return the original question as one item.

Original question:
{question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(DECOMPOSITION_TEMPLATE)


def decompose(main_question: str) -> list[str]:
    llm = build_llm()

    chain = decomposition_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": main_question})

    questions = []

    for line in result.splitlines():
        cleaned = line.strip()

        if not cleaned:
            continue

        # Removes formats such as: "1. Question" or "2) Question"
        if cleaned[0].isdigit():
            cleaned = cleaned.lstrip("0123456789. )-")

        if cleaned:
            questions.append(cleaned.strip())

    return questions[:3] or [main_question]

if __name__ == "__main__":
    result = decompose("What are the main components of an LLM-powered autonomous agent system?")
    for q in result:
        print(q)
