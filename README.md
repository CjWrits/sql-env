---
title: SQL Query Environment
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
  - sql
  - reinforcement-learning
  - spider
pinned: false
---

# SQL Query Environment

A real-world OpenEnv environment where AI agents learn to translate natural language questions into correct SQL queries — powered by the [Spider benchmark](https://huggingface.co/datasets/spider).

## Environment

```
reset(task_id)  →  question + database schema
step(query)     →  feedback + reward (0.0–1.0)
state()         →  episode metadata
```

## Tasks

| Task | Difficulty | SQL Complexity | Max Attempts |
|------|-----------|----------------|--------------|
| `easy` | Easy | Single table: WHERE, ORDER BY, COUNT | 3 |
| `medium` | Medium | Two-table JOIN | 4 |
| `hard` | Hard | Subquery, GROUP BY + HAVING, UNION | 5 |

## Action Space

```python
SQLAction(query: str)  # A valid SQL SELECT statement
```

## Observation Space

```python
SQLObservation(
    question, db_id, db_schema, task_id,
    attempt, max_attempts, last_query,
    last_error, feedback, done, reward
)
```

## Reward

| Situation | Reward |
|-----------|--------|
| Exact match | `1.0` |
| Partial match | `0.0–0.5` |
| Wrong result | `0.0` |
| Forbidden keyword | `-0.1` |

## API Endpoints

| Endpoint | Method |
|----------|--------|
| `/health` | GET |
| `/reset` | POST |
| `/step` | POST |
| `/state` | GET |
| `/tasks` | GET |
| `/grader` | POST |
| `/baseline` | POST |
| `/docs` | GET |

## Environment Variables (set in Space Settings → Variables)

- `API_BASE_URL` — LLM API endpoint
- `MODEL_NAME` — Model identifier
- `HF_TOKEN` — Your API key
