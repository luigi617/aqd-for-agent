import os
from typing import Type
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models.common import get_seed


def get_gemini(
    model_name: str,
    pydantic_model: Type[BaseModel],
    temperature: float = 0.1,
    timeout: int = 120,
):

    SEED = get_seed()
    if SEED is not None:
        temperature = 0.0

    base_llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        timeout=timeout,
        convert_system_message_to_human=True,
    )

    llm_structured = base_llm.with_structured_output(pydantic_model)
    base_prompt = ChatPromptTemplate.from_messages([("user", "{prompt}")])
    chain = base_prompt | llm_structured

    if chain is None:
        raise RuntimeError("LLM not available for the given model_name")

    return chain
