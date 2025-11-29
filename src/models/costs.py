# costs.py
from typing import Dict
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

OPENAI_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-5-mini": {"input": 0.00025, "cached_input": 0.000025, "output": 0.002},
    "gpt-5-nano": {"input": 0.00005, "cached_input": 0.000005, "output": 0.0004},
}

DEEPSEEK_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    "deepseek-chat": {"input": 0.00028, "cached_input": 0.000028, "output": 0.00042},
    "deepseek-reasoner": {"input": 0.00028, "cached_input": 0.000028, "output": 0.00042},
}

GEMINI_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash-lite": {"input": 0.0001, "cached_input": 0.00001, "output": 0.0004},
    "gemini-2.5-flash": {"input": 0.0003, "cached_input": 0.00003, "output": 0.0025},
}

OLLAMA_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    "qwen3:8b": {"input": 0.0, "cached_input": 0.0, "output": 0.0},
}

class TokenUsageCallback(BaseCallbackHandler):
    """Collect token usage from a single LLM call."""
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.cached_prompt_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        
        usage = (response.llm_output or {}).get("token_usage")
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

            prompt_details = usage.get("prompt_tokens_details") or {}
            self.cached_prompt_tokens += prompt_details.get("cached_tokens", 0)
            return
        
        # for gemini
        try:
            usage = response.generations[0][0].message.usage_metadata
            self.prompt_tokens += usage.get("input_tokens", 0)
            self.completion_tokens += usage.get("output_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

            prompt_details = usage.get("input_token_details") or {}
            self.cached_prompt_tokens += prompt_details.get("cache_read", 0)
            return
        except:
            pass


def _lookup_price(model_name: str) -> Dict[str, float]:
    if "gpt" in model_name:
        return OPENAI_PRICES_PER_1K[model_name]
    if "deepseek" in model_name:
        return DEEPSEEK_PRICES_PER_1K[model_name]
    if "qwen" in model_name or "llama" in model_name:
        return OLLAMA_PRICES_PER_1K[model_name]

    return {"input": 0.0, "cached_input": 0.0, "output": 0.0}


def compute_cost_usd(
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_prompt_tokens: int = 0,) -> float:
    prices = _lookup_price(model_name)
    return (
        ((prompt_tokens - cached_prompt_tokens) / 1000.0) * prices["input"]
        + (cached_prompt_tokens / 1000.0) * prices["cached_input"]
        + (completion_tokens / 1000.0) * prices["output"]
    )
