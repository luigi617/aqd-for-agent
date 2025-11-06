import os
from typing import Type
from pydantic import BaseModel

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from models.common import get_seed


def get_deepseek(
    model_name: str,
    pydantic_model: Type[BaseModel],
    temperature: float = 0.1,
    timeout: int = 120,
):
    
    SEED = get_seed()
    if SEED is not None:
        temperature = 0.0
    

    base_llm = ChatDeepSeek(
        model=model_name,
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