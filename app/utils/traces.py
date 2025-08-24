from typing import List, Dict, Any
from pydantic import BaseModel
from domain.candidate import Candidate

class StepTrace(BaseModel):
    """
    Represents the trace of a single step in a multi-step process, including model choices, candidates, and verification results.
    """
    step_idx: int
    skill: str
    chosen_models: List[str]
    candidates: List[Candidate]
    chosen_idx: int
    judge_meta: Dict[str, Any]
    verified: bool
    verifier_meta: Dict[str, Any]

class RunTrace(BaseModel):
    """
    Represents the trace of an entire run, including the task, final output, step traces, and overall cost and latency.
    """
    task: str
    final_text: str
    steps: List[StepTrace]
    total_cost_usd: float
    total_latency_s: float
