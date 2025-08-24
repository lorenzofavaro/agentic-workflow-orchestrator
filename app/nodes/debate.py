import asyncio
from typing import Dict, List
from adapter.adapter import CallRequest, LLMAdapter
from domain.candidate import Candidate

class Debate:
    """
    Orchestrates a debate among multiple language model adapters, running them in parallel and collecting their responses as candidates.
    
    Attributes:
        adapters (Dict[str, LLMAdapter]): Mapping from adapter names to LLMAdapter instances.
    """
    def __init__(self, adapters: Dict[str, LLMAdapter]):
        self.adapters = adapters

    async def run(self, names: List[str], req: CallRequest) -> List[Candidate]:
        """
        Run a debate among the specified adapters, collecting their responses in parallel.
        Args:
            names (List[str]): List of adapter names to participate in the debate.
            req (CallRequest): The request to send to each adapter.
        Returns:
            List[Candidate]: List of candidate responses from each adapter.
        """
        async def one(name: str) -> Candidate:
            res = await self.adapters[name].acomplete(req)
            return Candidate(
                name,
                res.text,
                res.latency_s,
                res.cost_usd,
                res.tokens_in,
                res.tokens_out,
            )

        return await asyncio.gather(*[one(n) for n in names])