# costs.py
from typing import Dict
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

OPENAI_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    "gpt-5-mini": {"input": 0.00025, "output": 0.002},
    "gpt-5-nano": {"input": 0.00005, "output": 0.0004},
}

DEEPSEEK_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    "deepseek-chat": {"input": 0.00028, "output": 0.00042},
    "deepseek-reasoner": {"input": 0.00028, "output": 0.00042},
}

OLLAMA_PRICES_PER_1K: Dict[str, Dict[str, float]] = {
    # local models – real dollar cost is "0", or put your internal estimate
    "qwen2.5:7b": {"input": 0.0, "output": 0.0},
    "llama3.1:8b": {"input": 0.0, "output": 0.0},
}

class TokenUsageCallback(BaseCallbackHandler):
    """Collect token usage from a single LLM call."""
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        # LangChain puts provider usage in llm_output["token_usage"]
        usage = (response.llm_output or {}).get("token_usage", {})  # type: ignore[attr-defined]

        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)


def _lookup_price(model_name: str) -> Dict[str, float]:
    # You can make this smarter if you have many aliases
    if "gpt" in model_name:
        return OPENAI_PRICES_PER_1K.get(model_name, {"input": 0.0, "output": 0.0})
    if "deepseek" in model_name:
        return DEEPSEEK_PRICES_PER_1K.get(model_name, {"input": 0.0, "output": 0.0})
    if "qwen" in model_name or "llama" in model_name:
        return OLLAMA_PRICES_PER_1K.get(model_name, {"input": 0.0, "output": 0.0})

    return {"input": 0.0, "output": 0.0}


def compute_cost_usd(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = _lookup_price(model_name)
    return (
        (prompt_tokens / 1000.0) * prices["input"]
        + (completion_tokens / 1000.0) * prices["output"]
    )
