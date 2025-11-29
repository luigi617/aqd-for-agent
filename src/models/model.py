from functools import lru_cache
from typing import Type, Optional
from time import sleep
from random import random
from json import JSONDecodeError

from pydantic import BaseModel, ValidationError
from openai import APIConnectionError, APITimeoutError, RateLimitError

from langchain_core.exceptions import OutputParserException
from src.models.deepseek import get_deepseek
from src.models.ollama import get_ollama
from src.models.gpt import get_openai



@lru_cache(maxsize=None)
def _get_base_llm(
    model_name: str,
    pydantic_model: Type[BaseModel],
    temperature: float,
    timeout: int,
    sensitive: bool,
):
    if "qwen" in model_name:
        return get_ollama(model_name, pydantic_model, temperature, timeout)

    if sensitive:
        raise ValueError("Cannot create API model with sensitive data")

    if "deepseek" in model_name:
        return get_deepseek(model_name, pydantic_model, temperature, timeout)
    elif "gpt" in model_name:
        return get_openai(model_name, pydantic_model, temperature, timeout)

    raise RuntimeError("LLM not available for the given model_name")

def get_model(
    model_name: str,
    pydantic_model: Type[BaseModel],
    temperature: float = 0.1,
    sensitive: bool = False,
    retries: int = 1, # -1 for infinite retries
    base_delay: float = 1.5,
    max_delay: float = 30.0,
    timeout: int = 120,
    all_fail_raise: bool = True,
):
    """
    Builds a JSON-only chain and returns a simple function:
        run(prompt_text: str) -> dict / pydantic model
    Every call to run() automatically retries on transient/parse errors.
    """

    chain = _get_base_llm(model_name, pydantic_model, temperature, timeout, sensitive)

    TRANSIENT_EXC = (APIConnectionError, APITimeoutError, RateLimitError)
    PARSE_EXC = (JSONDecodeError, OutputParserException, ValidationError)

    def run(prompt_text: str):
        # how many total tries (first try + (retries-1) retries)
        infinite = (retries == -1)
        max_tries = float("inf") if infinite else max(1, retries)
        delay = base_delay
        last_exc: Optional[BaseException] = None

        attempt = 0
        while attempt < max_tries:
            try:
                out = chain.invoke({"prompt": prompt_text})
                model_obj = out if isinstance(out, BaseModel) else pydantic_model.model_validate(out)
                return model_obj.model_dump()
            except (*TRANSIENT_EXC, *PARSE_EXC) as e:
                last_exc = e
            except Exception as e:
                last_exc = e
            
            attempt += 1
            if not infinite and attempt >= max_tries:
                if all_fail_raise and last_exc is not None:
                    raise last_exc
                return None

            sleep(delay + random() * 0.5)
            delay = min(delay * 2, max_delay)

    return run
