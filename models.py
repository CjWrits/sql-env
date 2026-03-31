"""
Data models for the SQL Query Environment.

An agent receives a natural language question + database schema,
and must write a correct SQL query to answer it.
"""

from typing import Any, Dict, List, Optional
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class SQLAction(Action):
    """The agent submits a SQL query string."""
    query: str = Field(..., description="SQL SELECT query to answer the question")


class SQLObservation(Observation):
    """What the agent sees: the question, schema, and feedback."""
    question: str = Field(default="", description="Natural language question to answer")
    db_id: str = Field(default="", description="Database identifier")
    db_schema: str = Field(default="", description="CREATE TABLE statements for all tables")
    task_id: str = Field(default="easy", description="Task difficulty: easy | medium | hard")
    attempt: int = Field(default=0, description="Current attempt number")
    max_attempts: int = Field(default=3, description="Maximum attempts allowed")
    last_query: Optional[str] = Field(default=None, description="Last submitted query")
    last_error: Optional[str] = Field(default=None, description="SQL error from last query, if any")
    feedback: Optional[str] = Field(default=None, description="Feedback on last attempt")


class SQLState(State):
    db_id: str = ""
    task_id: str = ""
    question: str = ""
    attempts_used: int = 0
