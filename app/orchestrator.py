import random

from adapter.adapter import CallRequest
from adapter.adapter import LLMAdapter
from domain.budget import Budget
from nodes.debate import Debate
from nodes.judge import Judge
from nodes.planner import Planner
from nodes.router import Router
from nodes.verifier import Verifier
from utils.prompts import AGENT_IMPROVE_SYSTEM_PROMPT
from utils.prompts import AGENT_IMPROVE_USER_PROMPT
from utils.prompts import AGENT_SYSTEM_PROMPT
from utils.traces import RunTrace
from utils.traces import StepTrace
import logging


class Orchestrator:
    """
    Orchestrates the execution of a multi-step plan using language model adapters, including debate, judging, and verification.
    """
    def __init__(
        self,
        adapters: dict[str, LLMAdapter],
        planner: Planner,
        router: Router,
        judge: Judge,
        verifier: Verifier,
        seed: int = 123,
    ):
        """
        Initialize the Orchestrator with all required components.
        Args:
            adapters (dict[str, LLMAdapter]): Mapping of adapter names to LLMAdapter instances.
            planner (Planner): The planner to generate plans.
            router (Router): The router to select adapters for each step.
            judge (Judge): The judge to select the best candidate.
            verifier (Verifier): The verifier to check candidate answers.
            seed (int): Random seed for reproducibility.
        """
        self.adapters = adapters
        self.planner = planner
        self.router = router
        self.judge = judge
        self.verifier = verifier
        self.rng = random.Random(seed)

    async def run(
        self, task: str, budget_usd: float, latency_s: float | None = None,
    ) -> RunTrace:
        """
        Execute the orchestrated process for a given task, budget, and latency constraint.
        Args:
            task (str): The task to solve.
            budget_usd (float): The hard budget limit in USD.
            latency_s (float | None): The hard latency limit in seconds (optional).
        Returns:
            RunTrace: Trace of the full run, including steps, costs, and outputs.
        """
        logging.info("[Orchestrator] Generating plan for task: '%s' (budget: %.2f USD)", task, budget_usd)
        plan = await self.planner.make_plan(task, budget_usd, latency_s)
        logging.info("[Orchestrator] Plan generated with %d steps.", len(plan.steps))
        budget = Budget(
            usd_left=plan.hard_budget_usd,
            deadline_s=plan.hard_latency_s,
        )
        total_cost = 0.0
        total_lat = 0.0
        step_traces: list[StepTrace] = []
        user_req = CallRequest(system=AGENT_SYSTEM_PROMPT, user=task)

        for si, step in enumerate(plan.steps):
            logging.info("[Step %d] Skill: %s | Description: %s", si + 1, step.skill, getattr(step, 'description', ''))
            names = self.router.pick_k(
                skill=step.skill, k=step.k_models, tier_hint=step.tier_hint,
            )
            logging.info("[Step %d] Selected models: %s", si + 1, names)
            # Debate (parallel)
            deb = Debate(self.adapters)
            logging.info("[Step %d] Running debate among selected models...", si + 1)
            cand = await deb.run(names, user_req)
            logging.info("[Step %d] Debate complete. Candidates: %d", si + 1, len(cand))
            step_cost = sum(c.cost_usd for c in cand)
            step_lat = max((c.latency_s for c in cand), default=0.0)
            if not budget.allow(step_cost, step_lat):
                logging.warning("[Step %d] Over budget/latency. Trimming to cheapest candidate.", si + 1)
                # If over budget, trim to cheapest single model
                cheapest = min(cand, key=lambda c: c.cost_usd)
                cand = [cheapest]
                step_cost = cheapest.cost_usd
                step_lat = cheapest.latency_s
            total_cost += step_cost
            total_lat += step_lat
            budget.charge(step_cost)

            # Judge
            logging.info("[Step %d] Judging candidates...", si + 1)
            if len(cand) == 1:
                j_idx = 0
                j_meta = {"judge_text": "There is only one candidate."}
            else:
                j_idx, j_meta = await self.judge.pick(task, cand)
            chosen = cand[j_idx]
            logging.info("[Step %d] Chosen candidate: #%d", si + 1, j_idx)

            # Verifier (optionally one improve round)
            logging.info("[Step %d] Verifying chosen candidate...", si + 1)
            ok, v_meta = await self.verifier.check(
                task, chosen.text, {'skill': step.skill},
            )
            verified = ok
            logging.info("[Step %d] Verification result: %s", si + 1, 'ACCEPTED' if ok else 'REJECTED')
            if (not ok) and step.max_rounds > 0 and budget.usd_left > 0.0:
                logging.info("[Step %d] Attempting improvement round...", si + 1)
                # Single improve round: escalate tier by +1 if available
                next_tier = max(self.adapters[n].spec.tier for n in names) + 1
                names2 = self.router.pick_k(
                    skill=step.skill, k=1, tier_hint=next_tier,
                )
                logging.info("[Step %d] Running improvement debate...", si + 1)
                cand2 = await Debate(self.adapters).run(
                    names2,
                    CallRequest(
                        system=AGENT_IMPROVE_SYSTEM_PROMPT,
                        user=AGENT_IMPROVE_USER_PROMPT.format(
                            task=task, previous=chosen.text,
                        ),
                    ),
                )
                logging.info("[Step %d] Improvement debate complete. Candidates: %d", si + 1, len(cand2))
                total_cost += sum(c.cost_usd for c in cand2)
                total_lat += max((c.latency_s for c in cand2), default=0.0)
                budget.charge(sum(c.cost_usd for c in cand2))
                # Judge between old and improved
                all_cand = [chosen] + cand2
                logging.info("[Step %d] Judging improved candidates...", si + 1)
                j_idx2, j_meta2 = await self.judge.pick(task, all_cand)
                chosen = all_cand[j_idx2]
                logging.info("[Step %d] Verifying improved candidate...", si + 1)
                ok2, v_meta2 = await self.verifier.check(
                    task, chosen.text, {'skill': step.skill, 'round': 2},
                )
                verified = ok2
                logging.info("[Step %d] Improvement verification result: %s", si + 1, 'ACCEPTED' if ok2 else 'REJECTED')
                # merge judge/verifier meta
                j_meta = {**j_meta, 'improve': j_meta2}
                v_meta = {**v_meta, 'improve': v_meta2}

            # Update router feedback with a very cheap binary reward (engineers can replace)
            self.router.update(
                chosen.model, reward=1.0 if verified else 0.0, cost=chosen.cost_usd,
            )

            step_traces.append(
                StepTrace(
                    step_idx=si,
                    skill=step.skill,
                    chosen_models=names,
                    candidates=cand,
                    chosen_idx=j_idx,
                    judge_meta=j_meta,
                    verified=verified,
                    verifier_meta=v_meta,
                ),
            )

        final_text = (
            step_traces[-1].candidates[step_traces[-1].chosen_idx].text
            if step_traces
            else ''
        )
        return RunTrace(
            task=task,
            final_text=final_text,
            steps=step_traces,
            total_cost_usd=total_cost,
            total_latency_s=total_lat,
        )
