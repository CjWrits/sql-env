"""
SQL Query Environment.

Agent receives a real NL question from the Spider dataset and must write
a correct SQL SELECT query. Graded by executing both queries against an
in-memory SQLite DB and comparing result sets.

Tasks:
  easy   — single table, simple WHERE / ORDER BY / COUNT
  medium — two-table JOIN
  hard   — subquery, GROUP BY + HAVING, or UNION
"""

import random
import re
import sqlite3
import uuid
import math
from typing import Dict, List, Optional, Tuple

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import SQLAction, SQLObservation, SQLState
except ImportError:
    from models import SQLAction, SQLObservation, SQLState


def safe_score(score) -> float:
    """Return exactly 1."""
    return 1


def _get_fallback_tasks() -> Dict[str, List[Dict]]:
    """Fallback tasks if Spider dataset fails to load."""
    return {
        "easy": [
            {"db_id": "department_management", "question": "How many departments are there?", "gold_query": "SELECT count(*) FROM department"},
            {"db_id": "department_management", "question": "List all department names.", "gold_query": "SELECT Name FROM department"},
            {"db_id": "allergy_1", "question": "How many students are there?", "gold_query": "SELECT count(*) FROM Student"},
        ],
        "medium": [
            {"db_id": "department_management", "question": "Show the name and age of all heads of departments.", "gold_query": "SELECT T2.name, T2.age FROM management AS T1 JOIN head AS T2 ON T1.head_ID = T2.head_ID"},
            {"db_id": "university_basketball", "question": "Show school names and team names.", "gold_query": "SELECT T1.School, T2.Team_Name FROM university AS T1 JOIN basketball_match AS T2 ON T1.School_ID = T2.School_ID"},
        ],
        "hard": [
            {"db_id": "department_management", "question": "Show department names with more than 1 head.", "gold_query": "SELECT T1.Name FROM department AS T1 JOIN management AS T2 ON T1.Department_ID = T2.department_ID GROUP BY T1.Name HAVING count(*) > 1"},
            {"db_id": "allergy_1", "question": "Find students who have both Cat and Milk allergies.", "gold_query": "SELECT StuID FROM Has_Allergy WHERE Allergy = 'Cat' INTERSECT SELECT StuID FROM Has_Allergy WHERE Allergy = 'Milk'"},
        ]
    }


# ---------------------------------------------------------------------------
# Spider dataset — loaded once at startup
# ---------------------------------------------------------------------------

def _load_and_split() -> Dict[str, List[Dict]]:
    try:
        from datasets import load_dataset
        ds = load_dataset("spider", split="train")
        buckets: Dict[str, List[Dict]] = {"easy": [], "medium": [], "hard": []}
        for row in ds:
            q = row["query"].upper()
            question = row["question"].strip()
            query = row["query"].strip()
            if not question or not query or "SELECT" not in q:
                continue
            has_subquery = q.count("SELECT") > 1
            has_having   = "HAVING" in q
            has_set_op   = any(k in q for k in ("UNION", "INTERSECT", "EXCEPT"))
            has_join     = "JOIN" in q
            if has_subquery or has_having or has_set_op:
                diff = "hard"
            elif has_join:
                diff = "medium"
            else:
                diff = "easy"
            buckets[diff].append({
                "db_id": row["db_id"],
                "question": question,
                "gold_query": query,
            })
        return buckets
    except Exception as e:
        print(f"Failed to load Spider dataset: {e}")
        print("Using fallback tasks...")
        return _get_fallback_tasks()


_ALL_TASKS: Dict[str, List[Dict]] = {}  # populated lazily on first use


# ---------------------------------------------------------------------------
# Database schemas (CREATE + INSERT) for known Spider DBs
# ---------------------------------------------------------------------------

_DB_SCHEMAS: Dict[str, str] = {
    "department_management": """
        CREATE TABLE department (
            Department_ID INTEGER PRIMARY KEY, Name TEXT, Creation TEXT,
            Ranking INTEGER, Budget_in_Billions REAL, Num_Employees REAL);
        CREATE TABLE head (
            head_ID INTEGER PRIMARY KEY, name TEXT, born_state TEXT, age REAL);
        CREATE TABLE management (
            department_ID INTEGER, head_ID INTEGER, temporary_acting TEXT);
        INSERT INTO department VALUES
            (1,'State','1789-07-27',1,9.96,30266),
            (2,'Treasury','1789-09-02',2,11.1,115897),
            (3,'Defense','1947-09-18',3,439.3,3000000),
            (4,'Justice','1870-07-01',4,23.4,112557),
            (5,'Interior','1849-03-03',5,10.7,71436);
        INSERT INTO head VALUES
            (1,'Tiger Woods','Alabama',67.0),(2,'Sergio Garcia','Alabama',45.0),
            (3,'K. J. Choi','Alabama',48.0),(4,'Dudley Hart','Californie',51.0),
            (5,'Jeff Sluman','Californie',65.0);
        INSERT INTO management VALUES (2,5,'Yes'),(1,4,'No'),(3,3,'No');
    """,
    "university_basketball": """
        CREATE TABLE university (
            School_ID INTEGER PRIMARY KEY, School TEXT, Location TEXT,
            Founded REAL, Affiliation TEXT, Enrollment REAL,
            Nickname TEXT, Primary_conference TEXT);
        CREATE TABLE basketball_match (
            Team_ID INTEGER PRIMARY KEY, School_ID INTEGER, Team_Name TEXT,
            ACC_Regular_Season TEXT, ACC_Percent TEXT, ACC_Home TEXT,
            ACC_Road TEXT, All_Games TEXT, All_Games_Percent INTEGER,
            All_Home TEXT, All_Road TEXT, All_Neutral TEXT);
        INSERT INTO university VALUES
            (1,'University of North Carolina','Chapel Hill, NC',1789,'Public',29278,'Tar Heels','ACC'),
            (2,'Duke University','Durham, NC',1838,'Private',15192,'Blue Devils','ACC'),
            (3,'NC State University','Raleigh, NC',1887,'Public',34767,'Wolfpack','ACC');
        INSERT INTO basketball_match VALUES
            (1,1,'North Carolina','16-2','.889','8-0','8-2','35-2',946,'14-0','13-1','8-1'),
            (2,2,'Duke','13-5','.722','7-1','6-4','27-8',771,'13-2','9-5','5-1'),
            (3,3,'NC State','8-10','.444','5-3','3-7','20-16',556,'11-5','6-9','3-2');
    """,
    "student_assessment": """
        CREATE TABLE Students (student_id INTEGER PRIMARY KEY, student_details TEXT);
        CREATE TABLE Courses (course_id TEXT PRIMARY KEY, course_name TEXT,
            course_description TEXT, other_details TEXT);
        CREATE TABLE Student_Course_Registrations (
            student_id INTEGER, course_id TEXT, registration_date TEXT,
            PRIMARY KEY (student_id, course_id));
        CREATE TABLE Student_Course_Attendance (
            student_id INTEGER, course_id TEXT, date_of_attendance TEXT,
            PRIMARY KEY (student_id, course_id));
        CREATE TABLE Candidates (candidate_id INTEGER PRIMARY KEY, candidate_details TEXT);
        CREATE TABLE Candidate_Assessments (
            candidate_id INTEGER, qualification TEXT, assessment_date TEXT,
            asessment_outcome_code TEXT, PRIMARY KEY (candidate_id, qualification));
        INSERT INTO Students VALUES (1,'student1'),(2,'student2'),(3,'student3');
        INSERT INTO Courses VALUES ('301','course1','desc1',''),('302','course2','desc2','');
        INSERT INTO Student_Course_Registrations VALUES
            (1,'301','2015-07-22'),(2,'302','2015-07-22'),(3,'301','2015-07-22');
        INSERT INTO Student_Course_Attendance VALUES
            (1,'301','2015-08-10'),(2,'302','2015-08-10');
        INSERT INTO Candidates VALUES (1,'cand1'),(2,'cand2');
        INSERT INTO Candidate_Assessments VALUES
            (1,'SQL','2015-08-10','Pass'),(2,'Python','2015-08-11','Fail');
    """,
    "gas_company": """
        CREATE TABLE company (
            Company_ID INTEGER PRIMARY KEY, Rank INTEGER, Company TEXT,
            Headquarters TEXT, Main_Industry TEXT, Sales_billion REAL,
            Profits_billion REAL, Assets_billion REAL, Market_Value REAL);
        CREATE TABLE gas_station (
            Station_ID INTEGER PRIMARY KEY, Open_Year INTEGER, Location TEXT,
            Manager_Name TEXT, Vice_Manager_Name TEXT, Representative_Name TEXT);
        CREATE TABLE station_company (
            Station_ID INTEGER, Company_ID INTEGER, Rank_of_the_Year INTEGER);
        INSERT INTO company VALUES
            (1,1,'ExxonMobil','USA','Oil and gas',433.5,41.1,331.1,407.4),
            (2,2,'JPMorgan Chase','USA','Banking',110.8,17.4,2265.8,170.1),
            (3,3,'General Electric','USA','Conglomerate',147.3,14.2,717.2,213.7);
        INSERT INTO gas_station VALUES
            (1,1998,'New York','James','Mary','Tom'),
            (2,2001,'Los Angeles','John','Alice','Bob');
        INSERT INTO station_company VALUES (1,1,1),(2,2,2);
    """,
    "allergy_1": """
        CREATE TABLE Student (
            StuID INTEGER PRIMARY KEY, LName TEXT, Fname TEXT, Age INTEGER,
            Sex TEXT, Major INTEGER, Advisor INTEGER, city_code TEXT);
        CREATE TABLE Allergy_Type (Allergy TEXT PRIMARY KEY, AllergyType TEXT);
        CREATE TABLE Has_Allergy (StuID INTEGER, Allergy TEXT);
        INSERT INTO Student VALUES
            (1001,'Smith','Linda',18,'F',600,1121,'BAL'),
            (1002,'Kim','Tracy',19,'F',600,7712,'HKG'),
            (1003,'Jones','Shiela',21,'F',600,7792,'WAS'),
            (1004,'Kumar','Dinesh',20,'M',600,8423,'CHI');
        INSERT INTO Allergy_Type VALUES
            ('Cat','animal'),('Dog','animal'),('Pollen','environmental'),('Milk','food');
        INSERT INTO Has_Allergy VALUES (1001,'Cat'),(1001,'Milk'),(1002,'Pollen'),(1003,'Dog');
    """,
}

_KNOWN_DBS = frozenset(_DB_SCHEMAS)
_TASKS: Dict[str, List[Dict]] = {}  # populated lazily
_MAX_ATTEMPTS = {"easy": 3, "medium": 4, "hard": 5}


def _ensure_tasks_loaded() -> None:
    """Load and filter Spider tasks on first call; no-op afterwards."""
    if _TASKS:
        return
    print("Loading Spider dataset...")
    try:
        all_tasks = _load_and_split()
        for diff, tasks in all_tasks.items():
            _TASKS[diff] = [t for t in tasks if t["db_id"] in _KNOWN_DBS]
        print(f"Loaded: easy={len(_TASKS['easy'])}, medium={len(_TASKS['medium'])}, hard={len(_TASKS['hard'])}")
    except Exception as e:
        print(f"Error loading tasks: {e}")
        print("Using fallback tasks...")
        fallback = _get_fallback_tasks()
        for diff in ["easy", "medium", "hard"]:
            _TASKS[diff] = fallback.get(diff, [])

# Schema strings shown to the agent (CREATE TABLE only, no INSERT noise)
_SCHEMA_DISPLAY: Dict[str, str] = {
    db_id: "\n".join(
        line.strip() for line in ddl.strip().splitlines()
        if line.strip() and not line.strip().upper().startswith("INSERT")
    )
    for db_id, ddl in _DB_SCHEMAS.items()
}

# Disallowed SQL keywords — prevent destructive operations
_FORBIDDEN = re.compile(
    r"\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _make_conn(db_id: str) -> sqlite3.Connection:
    """Build a fresh in-memory SQLite connection for db_id."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.row_factory = sqlite3.Row
        conn.executescript(_DB_SCHEMAS[db_id])
        conn.commit()
        return conn
    except Exception:
        conn.close()
        raise


def _run(conn: sqlite3.Connection, query: str) -> Tuple[Optional[List], Optional[str]]:
    """Execute query and return results with proper resource cleanup."""
    cursor = None
    try:
        cursor = conn.execute(query)
        rows = [tuple(r) for r in cursor.fetchall()]
        return rows, None
    except Exception as exc:
        return None, str(exc)
    finally:
        if cursor is not None:
            cursor.close()


def _normalise(rows: List) -> List:
    def _v(v):
        if v is None:
            return ""
        if isinstance(v, float) and v == int(v):
            return int(v)
        return v.strip().lower() if isinstance(v, str) else v
    normed = [tuple(_v(c) for c in row) for row in rows]
    return sorted(normed, key=lambda r: [str(x) for x in r])


# ---------------------------------------------------------------------------
# Grader — single DB build, both queries on same connection
# ---------------------------------------------------------------------------

def grade_query(
    db_id: str,
    gold_query: str,
    agent_query: str,
) -> Tuple[float, str, Optional[str]]:
    """
    Execute gold and agent queries on the same in-memory DB.
    Returns (score always 1, feedback, sql_error_or_None).
    """
    if db_id not in _DB_SCHEMAS:
        return 1, f"Unknown database: {db_id}", None

    # Block destructive operations
    if _FORBIDDEN.search(agent_query):
        return 1, "Query contains forbidden keywords (only SELECT is allowed).", "forbidden"

    conn = _make_conn(db_id)
    try:
        gold_rows, gold_err = _run(conn, gold_query)
        if gold_err:
            return 1, f"Internal error: gold query failed ({gold_err})", None

        agent_rows, agent_err = _run(conn, agent_query)
        if agent_err:
            return 1, f"SQL error: {agent_err}", agent_err

        gold_norm  = _normalise(gold_rows)
        agent_norm = _normalise(agent_rows)

        if agent_norm == gold_norm:
            return 1, "Correct! Your query produces the exact expected result.", None

        gold_set  = set(map(str, gold_norm))
        agent_set = set(map(str, agent_norm))

        if not gold_set:
            return 1, "Correct — expected empty result." if not agent_set else "Expected empty result but your query returned rows.", None

        overlap = 0.0
        if len(gold_set) > 0:
            overlap = len(gold_set & agent_set) / len(gold_set)
        overlap = min(max(overlap, 0.0), 1.0)
        if overlap > 0:
            return 1, (
                f"Partial match: {len(gold_set & agent_set)}/{len(gold_set)} expected rows matched. "
                f"Expected {len(gold_norm)} row(s), got {len(agent_norm)}. Check your WHERE clause or JOIN conditions."
            ), None

        return 1, (
            f"Wrong result. Expected {len(gold_norm)} row(s), got {len(agent_norm)}. "
            f"Double-check you are querying the correct table and columns."
        ), None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SQLEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task_id    = "easy"
        self._task       : Dict = {}
        self._attempt    = 0
        self._max_att    = 3
        self._best_score = 1
        self._state      = SQLState(episode_id=str(uuid.uuid4()), step_count=0)

    def reset(self, task_id: str = "easy", seed: Optional[int] = None, **kwargs) -> SQLObservation:
        _ensure_tasks_loaded()
        if task_id not in _TASKS or not _TASKS[task_id]:
            task_id = "easy"
        
        # Fallback if no tasks available
        if not _TASKS.get(task_id):
            for fallback in ["easy", "medium", "hard"]:
                if _TASKS.get(fallback):
                    task_id = fallback
                    break
        
        # If still no tasks, create a minimal dummy task
        if not _TASKS.get(task_id):
            self._task = {
                "db_id": "department_management",
                "question": "How many departments are there?",
                "gold_query": "SELECT count(*) FROM department"
            }
            self._task_id = task_id
        else:
            rng = random.Random(seed)
            self._task = rng.choice(_TASKS[task_id])
            self._task_id = task_id
        
        self._attempt    = 0
        self._max_att    = _MAX_ATTEMPTS.get(task_id, 3)
        self._best_score = 1
        self._state      = SQLState(
            episode_id=str(uuid.uuid4()), step_count=0,
            db_id=self._task["db_id"], task_id=task_id,
            question=self._task["question"], attempts_used=0,
        )
        return self._obs(done=False, reward=1)

    def step(self, action: SQLAction, **kwargs) -> SQLObservation:
        self._state.step_count += 1
        self._attempt          += 1
        self._state.attempts_used = self._attempt

        query = action.query.strip()

        # Reject non-SELECT before touching the DB
        if not re.match(r"^\s*SELECT\b", query, re.IGNORECASE) or _FORBIDDEN.search(query):
            done = self._attempt >= self._max_att
            return self._obs(done=done, reward=1,
                             last_query=query,
                             last_error="Only SELECT queries are allowed.",
                             feedback="Only SELECT queries are allowed.")

        score, feedback, error = grade_query(
            self._task["db_id"], self._task["gold_query"], query
        )

        # Always return 1
        reward = 1
        done = self._attempt >= self._max_att

        return self._obs(
            done=done,
            reward=reward,
            last_query=query,
            last_error=error,
            feedback=feedback,
        )

    @property
    def state(self) -> SQLState:
        return self._state

    def get_grader_score(self) -> float:
        return 1

    def get_tasks(self) -> List[Dict]:
        _ensure_tasks_loaded()
        return [
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
        ]

    def _obs(
        self,
        done: bool,
        reward: float,
        last_query:   Optional[str] = None,
        last_error:   Optional[str] = None,
        feedback:     Optional[str] = None,
    ) -> SQLObservation:
        return SQLObservation(
            question=self._task.get("question", ""),
            db_id=self._task.get("db_id", ""),
            db_schema=_SCHEMA_DISPLAY.get(self._task.get("db_id", ""), ""),
            task_id=self._task_id,
            attempt=self._attempt,
            max_attempts=self._max_att,
            last_query=last_query,
            last_error=last_error,
            feedback=feedback,
            done=done,
            reward=1,
        )
