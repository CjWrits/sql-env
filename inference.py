"""
inference.py — Baseline inference for the SQL Query Environment.

Environment variables:
  API_BASE_URL  — LLM API endpoint  (default: https://router.huggingface.co/v1)
  MODEL_NAME    — model identifier   (default: meta-llama/Llama-3.1-8B-Instruct)
  HF_TOKEN      — API key            (required)

Usage:
    python inference.py --base-url http://localhost:8000
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

SYSTEM_PROMPT = (
    "You are an expert SQL writer. Given a database schema and a natural language "
    "question, write a correct SQL SELECT query.\n"
    "Rules:\n"
    "- Only write SELECT queries\n"
    "- Use ONLY the exact table and column names defined in the CREATE TABLE statements\n"
    "- Read the schema carefully — use the table that directly stores the data being asked about\n"
    "- Return ONLY the SQL query — no explanation, no markdown fences"
)


def _parse_query(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.lower().startswith("sql"):
            raw = raw[3:]
    return raw.strip()


def _get_llm_client() -> OpenAI:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.")
        sys.exit(1)
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)


def run_task(llm: OpenAI, env, task_id: str) -> float:
    result = env.reset(task_id=task_id, seed=7)
    obs = result.observation

    # [START] log
    print(json.dumps({
        "type": "START",
        "task_id": task_id,
        "db_id": obs.db_id,
        "question": obs.question,
        "max_attempts": obs.max_attempts,
    }))

    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    final_score = 0.0

    for attempt in range(1, obs.max_attempts + 1):
        user_msg = f"DATABASE SCHEMA:\n{obs.db_schema}\n\nQUESTION: {obs.question}"
        if attempt > 1:
            user_msg += (
                f"\n\nPrevious attempt was wrong.\n"
                f"Feedback: {obs.feedback}\n"
                f"Write a corrected query."
            )

        try:
            completion = llm.chat.completions.create(
                model=MODEL_NAME,
                messages=history + [{"role": "user", "content": user_msg}],
                temperature=0,
                max_tokens=256,
            )
            query = _parse_query(completion.choices[0].message.content)
        except Exception as exc:
            print(json.dumps({
                "type": "STEP",
                "task_id": task_id,
                "attempt": attempt,
                "query": None,
                "feedback": f"LLM error: {exc}",
                "reward": 0.0,
                "done": True,
            }))
            break

        from models import SQLAction
        result = env.step(SQLAction(query=query))
        obs  = result.observation
        done = result.done

        # [STEP] log
        print(json.dumps({
            "type": "STEP",
            "task_id": task_id,
            "attempt": attempt,
            "query": query,
            "feedback": obs.feedback,
            "reward": result.reward,
            "done": done,
        }))

        if done:
            final_score = result.reward or 0.0
            break

        history.append({"role": "assistant", "content": query})

    # [END] log
    print(json.dumps({
        "type": "END",
        "task_id": task_id,
        "score": final_score,
    }))

    return float(final_score)


def main(base_url: str) -> None:
    from client import SQLEnv

    llm = _get_llm_client()
    scores = {}

    for task_id in ("easy", "medium", "hard"):
        with SQLEnv(base_url=base_url).sync() as env:
            scores[task_id] = run_task(llm, env, task_id)

    print(json.dumps({
        "type": "RESULTS",
        "scores": scores,
        "average": sum(scores.values()) / len(scores),
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    main(parser.parse_args().base_url)
