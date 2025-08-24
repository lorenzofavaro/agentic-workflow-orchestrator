
import asyncio
import logging

from orchestrator import Orchestrator
from domain.skill import Skill
from nodes.judge import LLMJudge
from nodes.planner import AgentPlanner
from nodes.router import Router
from nodes.verifier import LLMVerifier
from adapter.openai_adapter import OpenAIAdapter
from dotenv import load_dotenv


load_dotenv('.env')
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


async def main():
    query = input("Query: ")
    logging.info("Starting orchestration for user query...")
    trace = await orchestrator.run(query, budget_usd=0.10)
    logging.info("\n=== FINAL RESULT ===\n%s", trace.final_text)
    logging.info("Total cost: %.4f USD", trace.total_cost_usd)
    logging.info("Total latency: %.2f s", trace.total_latency_s)


if __name__ == '__main__':
    worker_models = {
        'gpt-4o': OpenAIAdapter('gpt-4o', cost_in=0.005, cost_out=0.015, tier=1, skills=(Skill.MATH, Skill.CODE)),
        'gpt-4o-mini': OpenAIAdapter('gpt-4o-mini', cost_in=0.001, cost_out=0.002, skills=(Skill.REASON, Skill.ANALYZE, Skill.SUMMARIZE)),
    }
    planner_model = OpenAIAdapter('gpt-4o', cost_in=0.005, cost_out=0.015)
    judge_model = OpenAIAdapter('gpt-4o', cost_in=0.005, cost_out=0.015)
    verifier_model = OpenAIAdapter('gpt-4o', cost_in=0.005, cost_out=0.015)

    orchestrator = Orchestrator(
        worker_models, AgentPlanner(planner_model), Router(
            worker_models,
        ), LLMJudge(judge_model), LLMVerifier(verifier_model),
    )
    asyncio.run(main())
