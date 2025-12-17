import math
import random
import threading
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

class UCB:
    def __init__(self, arms: Iterable[Any], variant: str = "ucb1", epsilon: float = 0.02):
        self.arms = list(arms)
        if not self.arms:
            raise ValueError("UCB requires at least one arm.")
        if variant not in ("ucb1", "ucb_tuned"):
            raise ValueError("variant must be 'ucb1' or 'ucb_tuned'")

        self.variant = variant
        self.epsilon = float(epsilon)

        self.counts: Dict[Any, int] = defaultdict(int)
        self.means: Dict[Any, float] = defaultdict(float)
        self.m2: Dict[Any, float] = defaultdict(float)
        self.total_pulls: int = 0

        self._lock = threading.Lock()
    
    def set_ucb(self, counts: Dict[Any,int], means: Dict[Any,float], m2: Dict[Any,float], total_pulls: int):
        with self._lock:
            self.counts = defaultdict(int, counts)
            self.means = defaultdict(float, means)
            self.m2 = defaultdict(float, m2)
            self.total_pulls = int(total_pulls)

    def select(self, k: int = 3, strategies_ignored = None, try_all: bool = False) -> List[Any]:
        if strategies_ignored is None:
            strategies_ignored = []

        if try_all:
            return [arm for arm in self.arms if arm not in strategies_ignored]
        
        k = max(0, min(k, len(self.arms)))
        if k <= 0: return []

        with self._lock:
            scored = self._score_arms(self.arms)
        scored.sort(key=lambda t: t[1], reverse=True)
        chosen = []
        idx = 0
        while len(chosen) < k and idx < len(scored):
            arm, _ = scored[idx]
            idx += 1
            if arm in strategies_ignored:
                continue
            if arm in chosen:
                continue
            chosen.append(arm)
        
        # If we didn't find enough (e.g., many ignored), fill randomly from remaining allowed arms
        if len(chosen) < k:
            remaining = [a for a in self.arms if a not in chosen and a not in strategies_ignored]
            random.shuffle(remaining)
            to_take = min(k - len(chosen), len(remaining))
            chosen.extend(remaining[:to_take])

        # epsilon exploration: occasionally swap one chosen with a random low-count arm
        if chosen and random.random() < self.epsilon:
            with self._lock:
                total = max(1, self.total_pulls)
                threshold = max(1, int(total * 0.05))
                low_count = [
                    arm for arm in self.arms
                    if self.counts[arm] <= threshold and arm not in chosen and arm not in strategies_ignored
                ]
            if low_count:
                replace_idx = -1  # replace last chosen by default
                chosen[replace_idx] = random.choice(low_count)

        return chosen

    def _score_arms(self, arms: Iterable[Any]) -> List[Tuple[Any, float]]:
        
        scored = []
        total = max(1, self.total_pulls)  # avoid log(0)
        log_total = math.log(total)

        for arm in arms:
            n = self.counts[arm]
            if n == 0:
                score = float("inf")
                scored.append((arm, score))
                continue

            mu = self.means[arm]
            if self.variant == "ucb1": # UCB1 score
                bonus = math.sqrt((2.0 * log_total) / n)
                score = mu + bonus
            else: # UCB1 tuned score
                var = (self.m2[arm] / n) if n > 1 else 0.0
                bonus_var = var + math.sqrt((2.0 * log_total) / n)
                bonus_var = min(0.25, bonus_var)
                bonus = math.sqrt((log_total / n) * bonus_var)
                score = mu + bonus
            scored.append((arm, score))

        return scored

    def update(self, arm: Any, reward: float) -> None:
        
        # incremental Welford update for mean and variance (M2)
        with self._lock:
            n = self.counts[arm] + 1
            prev_mean = self.means[arm]
            delta = reward - prev_mean
            new_mean = prev_mean + delta / n
            # update M2
            prev_m2 = self.m2[arm]
            new_m2 = prev_m2 + delta * (reward - new_mean)

            # store
            self.counts[arm] = n
            self.means[arm] = new_mean
            self.m2[arm] = new_m2

            self.total_pulls += 1

    def to_dict(self):
        with self._lock:
            return {
                "variant": self.variant,
                "epsilon": self.epsilon,
                "counts": dict(self.counts),
                "means": dict(self.means),
                "m2": dict(self.m2),
                "total_pulls": self.total_pulls,
            }
