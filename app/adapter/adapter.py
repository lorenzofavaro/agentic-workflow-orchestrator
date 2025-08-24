from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, Protocol
from pydantic import BaseModel
from domain.skill import Skill


@dataclass
class ModelSpec:
    """
    Specification for a language model, including provider, pricing, limits, and declared skills.
    """
    name: str
    provider: str  # e.g., "openai:gpt-4o", "anthropic:sonnet"
    cost_per_1k_input: float
    cost_per_1k_output: float
    rpm_limit: Optional[int] = None  # requests per minute
    tpm_limit: Optional[int] = None  # tokens per minute
    max_output_tokens: int = 512
    tier: int = 0  # 0=cheap, 1=mid, 2=premium
    skills: Tuple[Skill, ...] = ()  # declared strengths ("code","math","summarize")

@dataclass
class CallRequest:
    """
    Represents a request to an LLM, including prompts, temperature, stopping criteria, and metadata.
    """
    system: str
    user: str
    temperature: float = 0.2
    stop: Optional[Sequence[str]] = None
    max_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # plan/skill/constraints

@dataclass
class CallResult:
    """
    Result of an LLM call, including output text, token usage, latency, cost, and optional structured data.
    """
    text: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    cost_usd: float
    raw: Any = None  # provider payload if needed
    structured: Optional[BaseModel] = None  # parsed structured data if applicable

class LLMAdapter(Protocol):
    """
    Protocol for LLM adapters, defining the required interface for model completion.
    """
    spec: ModelSpec

    async def acomplete(self, req: CallRequest) -> CallResult:
        """
        Asynchronously complete a prompt using the LLM.
        Args:
            req (CallRequest): The request containing prompts and parameters.
        Returns:
            CallResult: The result of the LLM call.
        """
        ...

    async def acomplete_structured(self, req: CallRequest, base_model: BaseModel) -> CallResult:
        """
        Asynchronously complete a prompt and parse the output into a structured model.
        Args:
            req (CallRequest): The request containing prompts and parameters.
            base_model (BaseModel): The Pydantic model to parse the output into.
        Returns:
            CallResult: The result of the LLM call, with structured data if parsing is successful.
        """
        ...
