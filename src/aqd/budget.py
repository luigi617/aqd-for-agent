# budget.py
from pathlib import Path
from threading import Lock
from dataclasses import dataclass, field
from typing import List
import json


class BudgetExceededError(RuntimeError):
    pass

@dataclass
class TokenUsage:
    cached_prompt_tokens: int
    prompt_tokens: int
    completion_tokens: int
    model: str

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
        if delta_usd is None or delta_usd <= 0:
            return True

        with self._lock:
            self.history.append(token_usage)
            new_total = self.spent_usd + delta_usd
            if new_total > self.limit_usd:
                return False
            self.spent_usd = new_total
            return True
        
    # functions for pickle
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_lock"] 
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = Lock()

    def add_or_raise(self,
                     delta_usd: float,
                     cached_prompt_tokens: int,
                     prompt_tokens: int,
                     completion_tokens: int,
                     model: str) -> None:
        token_usage = TokenUsage(cached_prompt_tokens=cached_prompt_tokens,
                                 prompt_tokens=prompt_tokens,
                                 completion_tokens=completion_tokens,
                                 model=model)
        if not self._try_add(delta_usd, token_usage):
            raise BudgetExceededError(
                f"Budget exceeded: attempted spend {delta_usd:.6f}, "
                f"spent={self.spent_usd:.6f}, limit={self.limit_usd:.6f}"
            )

    def dump(self, path: Path):
        with open(path, 'w') as f:
            # use default to save TokenUsage object even though it's not JSON serializable
            json.dump({'spent_usd': self.spent_usd, 'history': self.history}, f, default=lambda o: o.__dict__, indent=4)