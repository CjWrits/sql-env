"""SQL Query Environment Client."""

from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from .models import SQLAction, SQLObservation, SQLState


class SQLEnv(EnvClient[SQLAction, SQLObservation, SQLState]):
    """
    Client for the SQL Query Environment.

    Example:
        >>> with SQLEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()          # easy task by default
        ...     print(result.observation.question)
        ...     print(result.observation.db_schema)
        ...     result = env.step(SQLAction(query="SELECT count(*) FROM head WHERE age > 56"))
        ...     print(result.observation.feedback)
        ...     print(result.reward)
    """

    def _step_payload(self, action: SQLAction) -> Dict:
        return {"query": action.query}

    def _parse_result(self, payload: Dict) -> StepResult[SQLObservation]:
        obs = payload.get("observation", {})
        observation = SQLObservation(
            question=obs.get("question", ""),
            db_id=obs.get("db_id", ""),
            db_schema=obs.get("db_schema", ""),
            task_id=obs.get("task_id", "easy"),
            attempt=obs.get("attempt", 0),
            max_attempts=obs.get("max_attempts", 3),
            last_query=obs.get("last_query"),
            last_error=obs.get("last_error"),
            feedback=obs.get("feedback"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SQLState:
        return SQLState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            db_id=payload.get("db_id", ""),
            task_id=payload.get("task_id", ""),
            question=payload.get("question", ""),
            attempts_used=payload.get("attempts_used", 0),
        )
