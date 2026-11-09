"""Shared LLM factory so every agent talks to the same model."""
import os

MODEL_NAME = "gemma-4-31b-it"


def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY before running the agent.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=MODEL_NAME, google_api_key=api_key)
