JUDGE_SYSTEM_PROMPT = "You are a strict evaluator."
JUDGE_USER_PROMPT = "Task: {task} Choose the best numbered answer and provide a short rationale.{listing}"
VERIFIER_SYSTEM_PROMPT = "You are a strict verifier."
VERIFIER_USER_PROMPT = "Task: {task} Answer: {answer} Metadata: {metadata}"
PLANNER_SYSTEM_PROMPT = "You are a useful planner."
PLANNER_PROMPT = """Create the **most concise step-by-step plan** to solve this task: {task}

Requirements:
- Minimize the number of steps; each step must be as short as possible.
- For each step, specify:
  1. **Skill**: Choose exactly one skill from: {skills}
  2. **Models (k_models)**: Specify as an **integer** the number of models that will compete to solve this step.
     - 1 for simple steps (only one model executes the step)
     - 2 for complex steps (two models with the required skill execute the step and the best result is chosen)
  3. **Tier hint**: Specify as an **integer** the minimum tier of model to consider for this step (0=cheap, 1=mid, 2=premium).
  4. **Max rounds**: Specify as an **integer** the maximum number of improvement rounds allowed by the verifier for this step (max 3 for really important steps).
  5. **Description**: Provide a short human-readable description of the step.

Format your response as a structured plan with brief, clear step descriptions only."""


AGENT_SYSTEM_PROMPT = "You are a helpful assistant."
AGENT_IMPROVE_SYSTEM_PROMPT = "Improve the answer."
AGENT_IMPROVE_USER_PROMPT = "Task: {task} Previous answer: {previous} Fix issues succinctly."