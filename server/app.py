"""
FastAPI app for the SQL Query Environment.

  GET  /tasks    — list tasks + action schema
  POST /grader   — score a query against a gold query
  POST /baseline — run LLM agent on all 3 tasks, return scores
"""

import os
import random
import logging
import re
import math

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)


def safe_score(score) -> float:
    """Ensure score is always strictly between 0 and 1."""
    try:
        score = float(score)
        if math.isnan(score) or math.isinf(score):
            return 0.02
    except (TypeError, ValueError):
        return 0.02
    return max(0.02, min(score, 0.95))

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("Install dependencies with 'uv sync'") from e

try:
    from models import SQLAction, SQLObservation
    from server.sql_env_environment import SQLEnvironment, grade_query, _TASKS, _SCHEMA_DISPLAY, _MAX_ATTEMPTS, _ensure_tasks_loaded
except ModuleNotFoundError:
    from ..models import SQLAction, SQLObservation
    from .sql_env_environment import SQLEnvironment, grade_query, _TASKS, _SCHEMA_DISPLAY, _MAX_ATTEMPTS, _ensure_tasks_loaded


app: FastAPI = create_app(
    SQLEnvironment, SQLAction, SQLObservation,
    env_name="sql_env", max_concurrent_envs=100,
)

@app.get("/tasks")
async def list_tasks():
    _ensure_tasks_loaded()
    return JSONResponse([
        {
            "task_id": tid,
            "difficulty": tid,
            "description": desc,
            "max_attempts": _MAX_ATTEMPTS[tid],
            "action_schema": {"query": "str — a valid SQL SELECT statement"},
            "example_question": _TASKS[tid][0]["question"] if _TASKS[tid] else "",
        }
        for tid, desc in {
            "easy":   "Single table SELECT with WHERE / ORDER BY / COUNT.",
            "medium": "JOIN across two tables.",
            "hard":   "Subquery, GROUP BY + HAVING, UNION, or INTERSECT.",
        }.items()
    ])


@app.post("/grader")
async def grader_score(request: Request):
    """
    Body: { "db_id": str, "gold_query": str, "agent_query": str }
    Returns: { "score": float, "feedback": str, "error": str|null }
    """
    body = await request.json()
    db_id       = body.get("db_id", "")
    gold_query  = body.get("gold_query", "")
    agent_query = body.get("agent_query", "")

    if not (db_id and gold_query and agent_query):
        return JSONResponse(
            {"error": "db_id, gold_query, and agent_query are all required"},
            status_code=400,
        )

    score, feedback, error = grade_query(db_id, gold_query, agent_query)
    return JSONResponse({"score": safe_score(score), "feedback": feedback, "error": error})


@app.post("/baseline")
async def baseline(request: Request):
    """
    Run LLM on one question per task (seed=42 for reproducibility).
    Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment.
    """
    hf_token     = os.environ.get("HF_TOKEN", "")
    api_base_url = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
    model_name   = os.environ.get("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

    if not hf_token:
        return JSONResponse({"error": "HF_TOKEN not set"}, status_code=400)

    body = {}
    try:
        body = await request.json()
    except ValueError as e:
        logging.warning(f"Invalid JSON in request body: {e}")
    except Exception as e:
        logging.exception(f"Unexpected error parsing request body: {e}")

    model = body.get("model", model_name)
    seed  = body.get("seed", 42)

    from openai import OpenAI
    client = OpenAI(api_key=hf_token, base_url=api_base_url)

    SYSTEM = (
        "You are an expert SQL writer. Given a schema and a question, "
        "return ONLY a valid SQL SELECT query — no explanation, no markdown."
    )

    results = {}
    rng = random.Random(seed)

    for task_id in ("easy", "medium", "hard"):
        tasks = _TASKS.get(task_id, [])
        if not tasks:
            results[task_id] = {"score": safe_score(0.02), "status": "no tasks available"}
            continue

        task   = rng.choice(tasks)
        schema = _SCHEMA_DISPLAY.get(task["db_id"], "")

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": (
                        f"DATABASE SCHEMA:\n{schema}\n\n"
                        f"QUESTION: {task['question']}"
                    )},
                ],
                temperature=0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
            
            # Sanitize LLM output - only allow SQL queries
            query = _sanitize_sql_output(raw)

            score, feedback, _ = grade_query(task["db_id"], task["gold_query"], query)
            results[task_id] = {
                "score": safe_score(score), "model": model, "status": "ok",
                "question": task["question"], "query": query, "feedback": feedback,
            }
        except ValueError as e:
            logging.error(f"Invalid LLM output for task {task_id}: {e}")
            results[task_id] = {"score": safe_score(0.02), "model": model, "status": f"validation_error: {str(e)}"}
        except Exception as exc:
            logging.exception(f"Error processing task {task_id}: {exc}")
            results[task_id] = {"score": safe_score(0.02), "model": model, "status": f"error: {str(exc)}"}

    return JSONResponse({"baseline_scores": results})


def _sanitize_sql_output(raw: str) -> str:
    """
    Sanitize LLM output to prevent injection attacks.
    Only allows valid SQL SELECT queries.
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


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point — call main() to start the server with defaults."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
