import random
from dataclasses import dataclass
from typing import Dict, List, Optional
from adapter.adapter import LLMAdapter


@dataclass
class RouterCfg:
    """
    Configuration for the router's selection policy.
    Args:
        epsilon (float): Probability of random exploration.
        alpha_cost (float): Weight for cost in the reward-cost tradeoff.
    """
    epsilon: float = 0.05  # exploration (for static policy)
    alpha_cost: float = 0.0  # tradeoff: reward − alpha·cost (if used)


@dataclass
class ArmStat:
    """
    Tracks statistics for an adapter (arm) in the router, including pulls, winrate, and cost.
    """
    pulls: int = 0
    winrate: float = 0.0
    cost: float = 0.0

    def update(self, reward: float, cost: float) -> None:
        """
        Update statistics with a new reward and cost observation.
        Args:
            reward (float): The observed reward.
            cost (float): The observed cost.
        """
        self.pulls += 1
        self.winrate += (reward - self.winrate) / self.pulls
        self.cost += (cost - self.cost) / self.pulls


class Router:
    """
    Selects language model adapters based on performance statistics and configurable policies, and updates their statistics after use.
    """
    def __init__(
        self,
        adapters: Dict[str, LLMAdapter],
        cfg: RouterCfg = RouterCfg(),
        seed: int = 123,
    ):
        """
        Initialize the Router with adapters, configuration, and random seed.
        Args:
            adapters (Dict[str, LLMAdapter]): Mapping of adapter names to LLMAdapter instances.
            cfg (RouterCfg): Router configuration.
            seed (int): Random seed for reproducibility.
        """
        self.adapters = adapters
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.stats: Dict[str, ArmStat] = {name: ArmStat() for name in adapters}

    def _score(self, name: str) -> float:
        """
        Compute a score for an adapter based on winrate, cost, and exploration.
        Args:
            name (str): The name of the adapter.
        Returns:
            float: The computed score.
        """
        s = self.stats[name]
        explore = self.rng.random() < self.cfg.epsilon
        if explore or s.pulls == 0:
            return self.rng.random()
        # Default static utility: winrate − alpha·cost
        return s.winrate - self.cfg.alpha_cost * s.cost

    def pick_k(self, skill: str, k: int, tier_hint: Optional[int] = None) -> List[str]:
        """
        Select up to k adapter names that match the skill and tier requirements, ordered by score.
        Args:
            skill (str): The required skill.
            k (int): Number of adapters to select.
            tier_hint (Optional[int]): Minimum model tier required.
        Returns:
            List[str]: List of selected adapter names.
        """
        cand = [
            a
            for a in self.adapters.values()
            if (tier_hint is None or a.spec.tier >= tier_hint)
            and (not skill or any(s.value == skill for s in a.spec.skills) or not a.spec.skills)
        ]
        if not cand:
            cand = list(self.adapters.values())
        scored = sorted(cand, key=lambda a: self._score(a.spec.name), reverse=True)
        return [a.spec.name for a in scored[: max(1, k)]]

    def update(self, name: str, reward: float, cost: float) -> None:
        """
        Update the statistics for a given adapter after an outcome.
        Args:
            name (str): The name of the adapter.
            reward (float): The observed reward.
            cost (float): The observed cost.
        """
        self.stats[name].update(reward, cost)