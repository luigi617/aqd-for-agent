import os
from typing import Type

from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from models.common import check_ollama_running, get_seed

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
    try:
        llm_structured = base_llm.with_structured_output(pydantic_model)
        base_prompt = ChatPromptTemplate.from_messages([("user", "{prompt}")])
        chain = base_prompt | llm_structured
    except Exception:
        parser = JsonOutputParser(pydantic_object=pydantic_model)
        SYSTEM_RULES = (
            "Return exactly one valid JSON object conforming to the schema below. "
            "No extra text, no code fences. Use null for unknowns.\n"
            "{format_instructions}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_RULES), ("user", "{prompt}")]
        ).partial(format_instructions=parser.get_format_instructions())
        chain = prompt | base_llm | parser
    
    if chain is None:
        raise RuntimeError("LLM not available for the given model_name")

    return chain