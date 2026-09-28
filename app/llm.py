import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def build_llm(
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        groq_api_key=GROQ_API_KEY,
    )
