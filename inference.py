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
import re
import math
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

BENCHMARK = "sql_env"


def safe_score(score) -> float:
    """Ensure score is always strictly between 0 and 1."""
    try:
        score = float(score)
        if math.isnan(score) or math.isinf(score):
            return 0.02
    except (TypeError, ValueError):
        return 0.02
    return max(0.02, min(score, 0.95))

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
    # Improve error reporting for better debugging
    if error and error != "null":
        error_val = error
    elif action == "null":
        error_val = "llm_error"
    elif reward < 0.1:
        error_val = "execution_error" if error else "low_score"
    else:
        error_val = "null"
    
    done_val = str(done).lower()
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
    """
    Parse and sanitize LLM output to extract SQL query.
    Validates output to prevent injection attacks.
    """
    raw = raw.strip()
    
    # Remove markdown code fences
    if raw.startswith("```"):
        parts = raw.split("```")
        if len(parts) > 1:
            raw = parts[1]
            if raw.lower().startswith("sql"):
                raw = raw[3:]
    
    query = raw.strip()
    
    # Validate: must start with SELECT
    if not re.match(r'^\s*SELECT\b', query, re.IGNORECASE):
        raise ValueError("Query must start with SELECT")
    
    # Block dangerous SQL keywords
    forbidden = re.compile(
        r'\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|REPLACE|'
        r'ATTACH|DETACH|PRAGMA|EXEC|EXECUTE|SCRIPT|LOAD_FILE|INTO\s+OUTFILE|'
        r'INTO\s+DUMPFILE|LOAD\s+DATA)\b',
        re.IGNORECASE
    )
    if forbidden.search(query):
        raise ValueError("Query contains forbidden SQL keywords")
    
    # Limit query length to prevent DoS
    if len(query) > 2000:
        raise ValueError("Query exceeds maximum length")
    
    # Remove any null bytes or control characters
    query = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', query)
    
    return query


def run_task(llm: OpenAI, env, task_id: str) -> float:
    from models import SQLAction

    result = env.reset(task_id=task_id)
    obs = result.observation

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.02  # Minimum valid score
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
                log_step(step=step, action="null", reward=safe_score(0.02), done=True, error=error)
                rewards.append(safe_score(0.02))
                steps_taken = step
                break

            result = env.step(SQLAction(query=query))
            obs  = result.observation
            done = result.done
            
            # Safe reward handling
            reward = safe_score(0.02)
            if isinstance(result.reward, (int, float)):
                reward = safe_score(result.reward)

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
                final_score = safe_score(reward)
                break

        success = final_score >= 0.90

    finally:
        # Ensure rewards list is never empty
        if not rewards:
            rewards = [safe_score(0.02)]
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return safe_score(final_score)


def main(base_url: str) -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.", flush=True)
        sys.exit(1)

    from client import SQLEnv

    llm = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    scores = {}

    for task_id in ("easy", "medium", "hard"):
        with SQLEnv(base_url=base_url).sync() as env:
            score = run_task(llm, env, task_id)
            
            # HARD SAFETY CLAMP (FINAL)
            try:
                score = float(score)
            except:
                score = 0.02
            
            if score <= 0.0:
                score = 0.02
            elif score >= 1.0:
                score = 0.95
            
            scores[task_id] = score

    # Safe average calculation with hard clamp
    avg_raw = sum(scores.values()) / len(scores) if scores else 0.02
    
    if avg_raw <= 0.0:
        avg = 0.02
    elif avg_raw >= 1.0:
        avg = 0.95
    else:
        avg = avg_raw
    
    print("=== Final Scores ===", flush=True)
    for tid, score in scores.items():
        print(f"  {tid:8s}: {score:.4f}", flush=True)
    print(f"  average : {avg:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    main(parser.parse_args().base_url)
