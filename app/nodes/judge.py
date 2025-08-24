from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Field
from adapter.adapter import CallRequest, CallResult, LLMAdapter
from domain.candidate import Candidate
import abc
from utils.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_PROMPT

class JudgeResponse(BaseModel):
    """
    Structured response from the judge, indicating the best answer and the reason for the choice.
    """
    best_answer_index: int = Field(..., description="The best answer'index among candidates")
    reason: str = Field(..., description="Short reason for the decision")

class Judge(abc.ABC):
    """
    Abstract base class for a judge that selects the best candidate from a list.
    """
    @abc.abstractmethod
    async def pick(
        self, task: str, candidates: List[Candidate]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select the best candidate for a given task.
        Args:
            task (str): The task or question to judge.
            candidates (List[Candidate]): List of candidate answers.
        Returns:
            Tuple[int, Dict[str, Any]]: Index of the best candidate and additional judge information.
        """
        ...

class LLMJudge(Judge):
    """
    Judge implementation that uses a language model adapter to select the best candidate answer.
    """
    def __init__(self, judge_adapter: LLMAdapter):
        """
        Initialize the LLMJudge with a given language model adapter.
        Args:
            judge_adapter (LLMAdapter): The language model adapter to use for judging.
        """
        self.judge = judge_adapter

    async def pick(
        self, task: str, candidates: List[Candidate]
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Use the language model to select the best candidate for the given task.
        Args:
            task (str): The task or question to judge.
            candidates (List[Candidate]): List of candidate answers.
        Returns:
            Tuple[int, Dict[str, Any]]: Index of the best candidate and additional judge information.
        """
        listing = "".join([f"[#{i}]{c.text}" for i, c in enumerate(candidates)])
        
        req = CallRequest(
            system=JUDGE_SYSTEM_PROMPT,
            user=JUDGE_USER_PROMPT.format(task=task, listing=listing),
        )
        res: CallResult = await self.judge.acomplete_structured(req, JudgeResponse)
        judge_response: JudgeResponse = res.structured
        return judge_response.best_answer_index, {"judge_text": str(judge_response)}
