import abc

from typing import List, Optional
from pydantic import BaseModel, Field
from utils.prompts import PLANNER_PROMPT
from domain.skill import Skill
from adapter.adapter import CallRequest, CallResult, LLMAdapter
from utils.prompts import PLANNER_SYSTEM_PROMPT


class PlanStep(BaseModel):
    """
    Represents a single step in a plan, including the required skill, description, and constraints.
    """
    skill: str = Field(..., description="The skill required for this step")
    description: str = Field(..., description="Human-readable description of the step")
    k_models: int = Field(2, description="Number of models that will compete for this step")
    tier_hint: Optional[int] = Field(None, description="Minimum model tier for this step: 0=cheap, 1=mid, 2=premium")
    max_rounds: int = Field(1, description="Number of improvement rounds allowed by the verifier")


class Plan(BaseModel):
    """
    Represents a multi-step plan, including steps, budget, latency, and random seed.
    """
    steps: List[PlanStep] = Field(..., description="List of plan steps")
    hard_budget_usd: float = Field(..., description="Hard budget limit in USD")
    hard_latency_s: Optional[float] = Field(None, description="Hard latency limit in seconds")
    seed: int = Field(123, description="Random seed for reproducibility")


class Planner(abc.ABC):
    """
    Abstract base class for planners that generate plans for a given task and constraints.
    """
    @abc.abstractmethod
    async def make_plan(
        self, task: str, budget_usd: float, latency_s: Optional[float]
    ) -> Plan:
        """
        Generate a plan for the given task and constraints.
        Args:
            task (str): The task to plan for.
            budget_usd (float): The hard budget limit in USD.
            latency_s (Optional[float]): The hard latency limit in seconds.
        Returns:
            Plan: The generated plan.
        """
        ...


class AgentPlanner(Planner):
    """
    Planner implementation that uses a language model adapter to generate plans.
    """
    def __init__(self, adapter: LLMAdapter, k: int = 2):
        """
        Initialize the AgentPlanner with a language model adapter and number of models per step.
        Args:
            adapter (LLMAdapter): The language model adapter to use for planning.
            k (int): Number of models to compete for each step (default: 2).
        """
        self.adapter = adapter
        self.k = k
        self.task_prompt = PLANNER_PROMPT

    async def make_plan(
        self, task: str, budget_usd: float, latency_s: Optional[float]
    ) -> Plan:
        """
        Generate a plan for the given task and constraints using the language model.
        Args:
            task (str): The task to plan for.
            budget_usd (float): The hard budget limit in USD.
            latency_s (Optional[float]): The hard latency limit in seconds.
        Returns:
            Plan: The generated plan.
        """
        user = self.task_prompt.format(task=task, skills=", ".join(skill.value for skill in Skill))
        req = CallRequest(system=PLANNER_SYSTEM_PROMPT, user=user, temperature=0, max_tokens=512)
        
        res: CallResult = await self.adapter.acomplete_structured(req, Plan)
        plan: Plan = res.structured
        plan.hard_budget_usd = budget_usd
        plan.hard_latency_s = latency_s
        
        return plan