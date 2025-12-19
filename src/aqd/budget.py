# budget.py
from threading import Lock
from dataclasses import dataclass, field
from typing import Any, List


class BudgetExceededError(RuntimeError):
    pass

@dataclass
class TokenUsage:
    cached_prompt_tokens: int
    prompt_tokens: int
    completion_tokens: int
    model: str
    budget_extra_info: Any

    def to_dict(self):
        return {
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            "budget_extra_info": self.budget_extra_info
        }

@dataclass
class Budget:
    """
    A simple, thread-safe shared budget.
    """
    limit_usd: float
    spent_usd: float = 0.0
    history: List[TokenUsage] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return max(self.limit_usd - self.spent_usd, 0.0)

    def _try_add(self, delta_usd: float, token_usage: TokenUsage) -> bool:
        """
        Thread-safe: atomically check if there is enough budget
        and subtract it. Returns True if successful, False if
        this would exceed the budget.
        """
        if delta_usd is None or delta_usd < 0:
            return True

        with self._lock:
            self.history.append(token_usage)
            new_total = self.spent_usd + delta_usd
            if new_total > self.limit_usd:
                return False
            self.spent_usd = new_total
            return True

    def add_or_raise(self,
                     delta_usd: float,
                     cached_prompt_tokens: int,
                     prompt_tokens: int,
                     completion_tokens: int,
                     model: str,
                     budget_extra_info: Any = None) -> None:
        token_usage = TokenUsage(cached_prompt_tokens=cached_prompt_tokens,
                                 prompt_tokens=prompt_tokens,
                                 completion_tokens=completion_tokens,
                                 model=model,
                                 budget_extra_info=budget_extra_info)
        if not self._try_add(delta_usd, token_usage):
            raise BudgetExceededError(
                f"Budget exceeded: attempted spend {delta_usd:.6f}, "
                f"spent={self.spent_usd:.6f}, limit={self.limit_usd:.6f}"
            )
        
    def to_dict(self):
        return {
            "limit_usd": self.limit_usd,
            "spent_usd": self.spent_usd,
            "history": [token_usage.to_dict() for token_usage in self.history]
        }