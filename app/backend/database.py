"""
PostgreSQL database module for PPE compliance tracking.

Schema:
- persons: Tracks unique individuals with first/last seen timestamps
- person_observations: Stores per-person PPE status observations over time
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime
from contextlib import contextmanager


# Database connection settings from environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "ppe_tracking")
DB_USER = os.getenv("DB_USER", "ppe_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "ppe_password")


def get_connection_string():
    """Get the PostgreSQL connection string."""
    return f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = psycopg2.connect(get_connection_string())
    try:
        yield conn
    finally:
        conn.close()


def init_database():
    """
    Initialize the database schema.
    Creates tables if they don't exist.
    Retries connection for Kubernetes (backend may start before PostgreSQL is ready).
    """
    import time

    max_retries = 10
    for attempt in range(max_retries):
        try:
            _init_schema()
            return
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"Database not ready (attempt {attempt + 1}/{max_retries}): {e}. Retrying in 3s..."
                )
                time.sleep(3)
            else:
                raise


def _init_schema():
    """Create tables if they don't exist."""
    with get_connection() as conn:
        cursor = conn.cursor()

        # Create persons table - tracks unique individuals
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                track_id INTEGER PRIMARY KEY,
                first_seen TIMESTAMP NOT NULL,
                last_seen TIMESTAMP NOT NULL
            )
        """)

        # Create person_observations table - stores PPE status per observation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS person_observations (
                id SERIAL PRIMARY KEY,
                track_id INTEGER NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                hardhat BOOLEAN,
                vest BOOLEAN,
                mask BOOLEAN,
                FOREIGN KEY (track_id) REFERENCES persons(track_id)
            )
        """)

        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_track_id
            ON person_observations(track_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_timestamp
            ON person_observations(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_hardhat
            ON person_observations(hardhat)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_vest
            ON person_observations(vest)
        """)

        conn.commit()
        print(f"PostgreSQL database initialized: {DB_HOST}:{DB_PORT}/{DB_NAME}")


# ----- Write Operations (used by tracker) -----


def insert_person(track_id: int, first_seen: datetime, last_seen: datetime):
    """Insert a new person record, or update last_seen if already exists."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO persons (track_id, first_seen, last_seen) VALUES (%s, %s, %s)
               ON CONFLICT (track_id) DO UPDATE SET last_seen = EXCLUDED.last_seen""",
            (track_id, first_seen, last_seen),
        )
        conn.commit()


def update_person_last_seen(track_id: int, last_seen: datetime):
    """Update the last_seen timestamp for an existing person."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE persons SET last_seen = %s WHERE track_id = %s",
            (last_seen, track_id),
        )
        conn.commit()


def insert_observation(
    track_id: int,
    timestamp: datetime,
    hardhat: bool = None,
    vest: bool = None,
    mask: bool = None,
):
    """Insert a new PPE observation record."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO person_observations (track_id, timestamp, hardhat, vest, mask)
               VALUES (%s, %s, %s, %s, %s)""",
            (track_id, timestamp, hardhat, vest, mask),
        )
        conn.commit()


# ----- Text-to-SQL Operations (used by chatbot) -----


def execute_query(sql: str) -> list:
    """
    Execute a SELECT query and return results as list of dicts.
    Used by the Text-to-SQL chatbot to run LLM-generated queries.

    Safety: Only SELECT queries are allowed. Dangerous keywords are blocked.
    """
    sql_upper = sql.strip().upper()

    # Only allow SELECT queries
    if not sql_upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    # Block dangerous keywords
    dangerous_keywords = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "GRANT",
        "REVOKE",
    ]
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            raise ValueError(f"Query contains forbidden keyword: {keyword}")

    with get_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(sql)
        results = cursor.fetchall()
        return [dict(row) for row in results]


def get_schema_description() -> str:
    """
    Return the database schema description for the LLM prompt.
    Used by the Text-to-SQL chatbot to understand the data model.
    """
    return """
DATABASE SCHEMA:

Table: persons
- track_id (INTEGER, PRIMARY KEY): Unique identifier for each tracked person
- first_seen (TIMESTAMP): When the person was first detected
- last_seen (TIMESTAMP): When the person was last detected

Table: person_observations
- id (SERIAL, PRIMARY KEY): Auto-incrementing observation ID
- track_id (INTEGER, FOREIGN KEY → persons.track_id): Links to the person
- timestamp (TIMESTAMP): When this observation was recorded
- hardhat (BOOLEAN): TRUE = wearing hardhat, FALSE = not wearing, NULL = not detected
- vest (BOOLEAN): TRUE = wearing safety vest, FALSE = not wearing, NULL = not detected
- mask (BOOLEAN): TRUE = wearing mask, FALSE = not wearing, NULL = not detected

NOTES:
- A "violation" means the person was NOT wearing required PPE (hardhat=FALSE, vest=FALSE, mask=FALSE)
- Use COUNT(DISTINCT track_id) to count unique people
- Use CURRENT_TIMESTAMP for current time, CURRENT_DATE for today
- Use INTERVAL for date math: CURRENT_DATE - INTERVAL '7 days'
- Use EXTRACT(DOW FROM timestamp) for day of week (0=Sunday, 6=Saturday)
- Use DATE_TRUNC('day', timestamp) to group by date
- Use TO_CHAR(timestamp, 'Day') to get day name
"""
