---
title: SQL Query Environment



sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
  - sql
  - reinforcement-learning
  - spider
pinned: false
---

# SQL Query Environment

A real-world OpenEnv environment where AI agents learn to translate natural language questions into correct SQL queries — powered by the [Spider benchmark](https://huggingface.co/datasets/spider).

## Motivation

Writing SQL from natural language is one of the most common tasks in data engineering and analytics. This environment lets agents practice and be evaluated on progressively harder SQL tasks with fully deterministic, execution-based grading.

## Environment

```
reset(task_id)  →  question + database schema
step(query)     →  feedback + reward (0.0–1.0)
state()         →  episode metadata
```

## Action Space

```python
SQLAction(query: str)  # A valid SQL SELECT statement
```

## Observation Space

```python
SQLObservation(
    question:     str        # Natural language question
    db_id:        str        # Database identifier
    db_schema:    str        # CREATE TABLE statements
    task_id:      str        # easy | medium | hard
    attempt:      int        # Current attempt number
    max_attempts: int        # Maximum attempts allowed
    last_query:   str|None   # Last submitted query
    last_error:   str|None   # SQL error if any
    feedback:     str|None   # Grader feedback
    done:         bool
    reward:       float|None
)
```

## Tasks

| Task | Difficulty | SQL Complexity | Max Attempts |
|------|-----------|----------------|--------------|
| `easy` | Easy | Single table: WHERE, ORDER BY, COUNT | 3 |
| `medium` | Medium | Two-table JOIN | 4 |
| `hard` | Hard | Subquery, GROUP BY + HAVING, UNION | 5 |

## Reward Function

| Situation | Reward |
|-----------|--------|
| Exact result match | `1.0` |
| Partial match | `0.0–0.5` |
| Wrong result | `0.0` |
| Forbidden keyword | `-0.1` |

## Baseline Scores

Scores using `llama-3.1-8b-instant`, `temperature=0`, `seed=7`:

| Task | Score |
|------|-------|
| easy | 1.0 |
| medium | 1.0 |
| hard | 1.0 |
| average | 1.0 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start episode |
| `/step` | POST | Submit query |
| `/state` | GET | Episode metadata |
| `/tasks` | GET | List all tasks |
| `/grader` | POST | Score any query |
| `/baseline` | POST | Run LLM baseline |
| `/docs` | GET | OpenAPI docs |

## Setup

```bash
# Install dependencies
pip install openenv-core datasets python-dotenv openai fastapi uvicorn

# Set environment variables
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export HF_TOKEN=your_token_here

# Run server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference
python inference.py --base-url http://localhost:8000
```

## Docker

```bash
docker build -t sql-env:latest -f server/Dockerfile .
docker run -p 8000:8000 \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
  -e HF_TOKEN=your_token \
  sql-env:latest
```

## Project Structure

```
sql_env/
├── inference.py               ← Baseline inference script (hackathon required)
├── openenv.yaml               ← OpenEnv manifest
├── pyproject.toml             ← Package metadata
├── README.md
├── models.py                  ← Typed Pydantic models
├── client.py                  ← WebSocket/HTTP client
└── server/
    ├── sql_env_environment.py ← Core environment logic
    ├── app.py                 ← FastAPI server
    └── Dockerfile             ← Container definition
```
