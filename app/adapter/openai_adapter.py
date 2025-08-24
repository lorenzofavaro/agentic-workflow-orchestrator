import os

from openai import AsyncOpenAI
from pydantic import BaseModel

from adapter.adapter import CallRequest
from adapter.adapter import CallResult
from adapter.adapter import LLMAdapter
from adapter.adapter import ModelSpec


class OpenAIAdapter(LLMAdapter):
    def __init__(self, model: str, cost_in: float, cost_out: float, tier: int = 0, skills: tuple = None):
        self.spec = ModelSpec(
            name=model,
            provider='openai',
            cost_per_1k_input=cost_in,
            cost_per_1k_output=cost_out,
            tier=tier,
            skills=skills,
        )
        self.aclient = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])

    async def acomplete(self, req: CallRequest) -> CallResult:
        import time
        t0 = time.time()

        resp = await self.aclient.chat.completions.create(
            model=self.spec.name,
            messages=[
                {'role': 'system', 'content': req.system},
                {'role': 'user', 'content': req.user},
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens or self.spec.max_output_tokens,
            stop=req.stop,
        )
        t1 = time.time()

        text = resp.choices[0].message.content
        tokens_in = resp.usage.prompt_tokens
        tokens_out = resp.usage.completion_tokens
        cost = (
            tokens_in / 1000 * self.spec.cost_per_1k_input +
            tokens_out / 1000 * self.spec.cost_per_1k_output
        )

        return CallResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=t1 - t0,
            cost_usd=cost,
            raw=resp,
        )

    async def acomplete_structured(self, req: CallRequest, base_model: BaseModel) -> CallResult:
        res = await self.acomplete(req=req)
        if res.text:
            try:
                structured_response = await self.aclient.responses.parse(
                    model=self.spec.name,
                    input=[
                        {
                            'role': 'system',
                            'content': 'Extract structured data from the following text.',
                        },
                        {'role': 'user', 'content': res.text},
                    ],
                    text_format=base_model,
                )
                if structured_response:
                    res.structured = structured_response.output_parsed
            except Exception as e:
                print(f'Error parsing text: {res.text}\n{e}')
        return res