import os
from typing import Type

from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.models.common import check_ollama_running, get_seed

IS_OLLAMA_RUNNING = False



def get_ollama(
    model_name: str,
    pydantic_model: Type[BaseModel],
    temperature: float = 0.1,
    timeout: int = 120,
):
    global IS_OLLAMA_RUNNING

    if not IS_OLLAMA_RUNNING:
        check_ollama_running(model_name)
    
    IS_OLLAMA_RUNNING = True

    SEED = get_seed()
    if SEED is not None:
        temperature = 0.0

    base_llm = ChatOllama(
        model=model_name,
        format="json",
        temperature=temperature,
        timeout=timeout,
        seed=SEED,
    )

    llm_structured = base_llm.with_structured_output(pydantic_model)
    base_prompt = ChatPromptTemplate.from_messages([("user", "{prompt}")])
    chain = base_prompt | llm_structured
    
    if chain is None:
        raise RuntimeError("LLM not available for the given model_name")

    return chain