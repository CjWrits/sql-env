"""
Inference Script — SQL Query Environment
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL      The API endpoint for the LLM.
    MODEL_NAME        The model identifier to use for inference.
    HF_TOKEN          Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

Usage:
    python inference.py --base-url http://localhost:8000
"""

import argparse
import os
import sys
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

BENCHMARK = "sql_env"

SYSTEM_PROMPT = (
    "You are an expert SQL writer. Given a database schema and a natural language "
    "question, write a correct SQL SELECT query.\n"
    "Rules:\n"
    "- Only write SELECT queries\n"
    "- Use ONLY the exact table and column names defined in the CREATE TABLE statements\n"
    "- Read the schema carefully — use the table that directly stores the data being asked about\n"
    "- Return ONLY the SQL query — no explanation, no markdown fences"
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # keep action on one line, truncate if needed
    action_clean = action.replace("\n", " ").strip()
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def _parse_query(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("sql"):
            raw = raw[3:]
    return raw.strip()


def run_task(llm: OpenAI, env, task_id: str) -> float:
    from models import SQLAction

    result = env.reset(task_id=task_id, seed=7)
    obs = result.observation

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False

    try:
        for step in range(1, obs.max_attempts + 1):
            if result.done:
                break

            user_msg = f"DATABASE SCHEMA:\n{obs.db_schema}\n\nQUESTION: {obs.question}"
            if step > 1:
                user_msg += (
                    f"\n\nPrevious attempt was wrong.\n"
                    f"Feedback: {obs.feedback}\n"
                    f"Write a corrected query."
                )

            error = None
            query = ""
            try:
                completion = llm.chat.completions.create(
                    model=MODEL_NAME,
                    messages=history + [{"role": "user", "content": user_msg}],
                    temperature=0,
                    max_tokens=256,
                )
                query = _parse_query(completion.choices[0].message.content)
            except Exception as exc:
                error = str(exc)
                log_step(step=step, action="null", reward=0.00, done=True, error=error)
                rewards.append(0.0)
                steps_taken = step
                break

            result = env.step(SQLAction(query=query))
            obs  = result.observation
            done = result.done
            reward = result.reward or 0.0

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=query,
                reward=reward,
                done=done,
                error=obs.last_error,
            )

            history.append({"role": "assistant", "content": query})

            if done:
                final_score = reward
                break

        success = final_score >= 1.0

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return float(final_score)


def main(base_url: str) -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.", flush=True)
        sys.exit(1)

    from client import SQLEnv

    llm = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    scores = {}

    for task_id in ("easy", "medium", "hard"):
        with SQLEnv(base_url=base_url).sync() as env:
            scores[task_id] = run_task(llm, env, task_id)

    avg = sum(scores.values()) / len(scores)
    print("=== Final Scores ===", flush=True)
    for tid, score in scores.items():
        print(f"  {tid:8s}: {score:.4f}", flush=True)
    print(f"  average : {avg:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    main(parser.parse_args().base_url)
