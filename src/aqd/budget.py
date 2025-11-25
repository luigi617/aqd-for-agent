# budget.py
from threading import Lock
from dataclasses import dataclass, field


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class Budget:
    """
    A simple, thread-safe shared budget.
    """
    limit_usd: float
    spent_usd: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def remaining_usd(self) -> float:
        with self._lock:
            return max(self.limit_usd - self.spent_usd, 0.0)

    def try_add(self, delta_usd: float) -> bool:
        """
        Thread-safe: atomically check if there is enough budget
        and subtract it. Returns True if successful, False if
        this would exceed the budget.
        """
        if delta_usd is None or delta_usd <= 0:
            return True

        with self._lock:
            new_total = self.spent_usd + delta_usd
            if new_total > self.limit_usd:
                return False
            self.spent_usd = new_total
            return True

    def add_or_raise(self, delta_usd: float) -> None:
        """
        Same as try_add, but raises BudgetExceededError if insufficient.
        """
        print(self.spent_usd, self.limit_usd, self.spent_usd / self.limit_usd)
        if not self.try_add(delta_usd):
            raise BudgetExceededError(
                f"Budget exceeded: attempted spend {delta_usd:.6f}, "
                f"spent={self.spent_usd:.6f}, limit={self.limit_usd:.6f}"
            )
