# budget.py
from threading import Lock
from dataclasses import dataclass, field
from contextvars import ContextVar
from contextlib import contextmanager


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class Budget:
    limit_usd: float
    spent_usd: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def remaining_usd(self) -> float:
        return max(self.limit_usd - self.spent_usd, 0.0)

    def add(self, delta_usd: float) -> None:
        """Add cost. Raise if this call pushes us over the limit."""
        if delta_usd is None:
            return
        with self._lock:
            new_total = self.spent_usd + delta_usd
            self.spent_usd = new_total
            if new_total > self.limit_usd:
                raise BudgetExceededError(
                    f"Budget exceeded: spent={new_total:.6f} > limit={self.limit_usd:.6f}"
                )


CURRENT_BUDGET: ContextVar[Budget | None] = ContextVar(
    "CURRENT_BUDGET", default=None
)


@contextmanager
def llm_budget(limit_usd: float) -> Budget:
    """
    Usage:

        with llm_budget(0.10) as budget:
            res1 = run_model1("...")
            res2 = run_model2("...")

    All get_model() calls inside will respect the same budget.
    """
    budget = Budget(limit_usd=limit_usd)
    token = CURRENT_BUDGET.set(budget)
    try:
        yield budget
    finally:
        CURRENT_BUDGET.reset(token)

def get_current_budget() -> Budget:
    """Convenience helper that guarantees a Budget or raises."""
    budget = CURRENT_BUDGET.get()
    if budget is None:
        raise RuntimeError("No active llm_budget context.")
    return budget