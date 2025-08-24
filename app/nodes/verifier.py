from typing import Any, Dict, Tuple
from pydantic import BaseModel, Field
from adapter.adapter import CallRequest, CallResult, LLMAdapter
import abc
from utils.prompts import VERIFIER_SYSTEM_PROMPT, VERIFIER_USER_PROMPT


class VerifyResponse(BaseModel):
    """
    Structured response from the verifier, indicating whether to accept or revise the answer and the reason.
    """
    response: str = Field(..., description="ACCEPT or REVISE the answer")
    reason: str = Field(..., description="Short reason for the decision")

class Verifier(abc.ABC):
    """
    Abstract base class for verifiers that check the validity of an answer for a given task.
    """
    @abc.abstractmethod
    async def check(
        self, task: str, answer: str, metadata: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check the validity of an answer for a given task and metadata.
        Args:
            task (str): The task or question being verified.
            answer (str): The answer to verify.
            metadata (Dict[str, Any]): Additional metadata for verification.
        Returns:
            Tuple[bool, Dict[str, Any]]: (True if accepted, False if revision needed; additional verifier info)
        """
        ...

class LLMVerifier(Verifier):
    """
    Verifier implementation that uses a language model adapter to check answers.
    """
    def __init__(self, verifier_adapter: LLMAdapter):
        """
        Initialize the LLMVerifier with a language model adapter.
        Args:
            verifier_adapter (LLMAdapter): The language model adapter to use for verification.
        """
        self.ver = verifier_adapter

    async def check(
        self, task: str, answer: str, metadata: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Use the language model to check the validity of an answer for a given task and metadata.
        Args:
            task (str): The task or question being verified.
            answer (str): The answer to verify.
            metadata (Dict[str, Any]): Additional metadata for verification.
        Returns:
            Tuple[bool, Dict[str, Any]]: (True if accepted, False if revision needed; additional verifier info)
        """
        req = CallRequest(
            system=VERIFIER_SYSTEM_PROMPT,
            user=VERIFIER_USER_PROMPT.format(task=task, answer=answer, metadata=metadata),
        )
        res: CallResult = await self.ver.acomplete_structured(req, VerifyResponse)
        verify_response: VerifyResponse = res.structured
        return verify_response.response == "ACCEPT", {"verifier_text": str(verify_response)}
