import json
import os

import google.generativeai as genai
from pydantic import BaseModel

from adapter.adapter import CallRequest
from adapter.adapter import CallResult
from adapter.adapter import LLMAdapter
from adapter.adapter import ModelSpec

class GoogleAdapter(LLMAdapter):
    def __init__(self, model: str, cost_in: float, cost_out: float, tier: int = 0, skills: tuple = None):
        self.spec = ModelSpec(
            name=model,
            provider='google',
            cost_per_1k_input=cost_in,
            cost_per_1k_output=cost_out,
            tier=tier,
            skills=skills,
        )
        genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
        self.model = genai.GenerativeModel(model)

    async def acomplete(self, req: CallRequest) -> CallResult:
        import time
        t0 = time.time()

        prompt = f'{req.system}\n\nUser: {req.user}'
        response = await self.model.generate_content_async(
            prompt,
            generation_config={
                'temperature': req.temperature,
                'max_output_tokens': req.max_tokens or self.spec.max_output_tokens,
            },
        )
        t1 = time.time()

        text = response.text
        tokens_in = response.usage_metadata.prompt_token_count
        tokens_out = response.usage_metadata.candidates_token_count
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
            raw=response,
        )

    async def acomplete_structured(self, req: CallRequest, base_model: BaseModel) -> CallResult:
        import time
        t0 = time.time()

        prompt = f'{req.system}\n\nUser: {req.user}'
        response = await self.model.generate_content_async(
            prompt,
            generation_config={
                'temperature': req.temperature,
                'max_output_tokens': req.max_tokens or self.spec.max_output_tokens,
                'response_mime_type': 'application/json',
                'response_schema': list[base_model],
            },
        )
        t1 = time.time()

        text = response.text
        tokens_in = response.usage_metadata.prompt_token_count
        tokens_out = response.usage_metadata.candidates_token_count
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
            raw=response,
            structured=base_model.model_validate(json.loads(response.text)[0]),
        )