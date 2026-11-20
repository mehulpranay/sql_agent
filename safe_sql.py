"""
safe_sql.py  --  everything that decides whether a model-written SQL query may touch a database.

Three independent layers. Each one still protects us if another has a hole:

    1. check_read_only()    reads the SQL as a syntax tree and rejects anything that is not
                            ONE plain read query. (Stops the query before it reaches the database.)
    2. connect_read_only()  opens the database file read-only, so the database itself refuses
                            writes even if layer 1 missed something.
                            It also installs an "authorizer": SQLite asks it before every
                            operation, and it only says yes to reading.
    3. resolve_db_path()    makes sure a db_id can only ever point at a file inside the
                            databases folder (a db_id like "../../secrets" must not work).

Layer 2 also carries the time limit: SQLite has no built-in "stop after N seconds", so we ask
it to call us every few thousand steps and tell it to abort once the time is up.
"""

import os
import time
import sqlite3
from pathlib import Path

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError


class UnsafeSQLError(ValueError):
    """Raised when a query or a db_id is not allowed. The message is safe to show to the model."""


# ----------------------------------------------------------------------------
# layer 1: look at the SQL before running it
# ----------------------------------------------------------------------------

# Anything that writes, changes structure, attaches another file or changes settings.
_FORBIDDEN_NODE_NAMES = (
    'Insert', 'Update', 'Delete', 'Merge', 'Create', 'Drop', 'Alter', 'TruncateTable',
    'Command', 'Pragma', 'Attach', 'Detach', 'Transaction', 'Commit', 'Rollback', 'Use',
)
_FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n))

# SQLite functions that can load code or touch files.
_FORBIDDEN_FUNCTIONS = {'load_extension', 'readfile', 'writefile', 'edit', 'fts3_tokenizer'}


def check_read_only(sql: str) -> None:
    """Return None if `sql` is a single read-only query, otherwise raise UnsafeSQLError."""

    if not sql or not sql.strip():
        raise UnsafeSQLError('Empty query.')

    try:
        statements = [s for s in sqlglot.parse(sql, read='sqlite') if s is not None] # sqlglot.parse() sees the semicolon and return a list of two AST trees, if one semicolon is present
    except SqlglotError as e:
        # A query we cannot even parse is a query we cannot vouch for.
        raise UnsafeSQLError(f'Could not parse the SQL: {str(e)[:200]}')

    if len(statements) != 1:
        raise UnsafeSQLError('Only one SQL statement is allowed per call.')

    stmt = statements[0]

    # Top level must be a query: SELECT, UNION/INTERSECT/EXCEPT of queries, or a bracketed query.
    # (WITH ... SELECT is parsed as a SELECT.)
    if not isinstance(stmt, exp.Query):
        raise UnsafeSQLError(f'Only SELECT queries are allowed (got {type(stmt).__name__.upper()}).')

    # Look at every part of the query, including subqueries.
    for node in stmt.walk():
        if isinstance(node, _FORBIDDEN_NODES):
            raise UnsafeSQLError(f'{type(node).__name__.upper()} is not allowed inside a query.')
        if isinstance(node, exp.Anonymous) and node.name.lower() in _FORBIDDEN_FUNCTIONS:
            raise UnsafeSQLError(f'The function {node.name} is not allowed.')

"""
Query: SELECT update_count FROM users WHERE status = 'DELETE'
└── Select (Allowed)
    ├── Column: "update_count" (Allowed - it's just a name)
    ├── Table: "users" (Allowed)
    └── Where (Allowed)
        └── Literal String: 'DELETE' (Allowed - it's just data)

Query: DELETE FROM users WHERE status = 'INACTIVE'
└── Delete (FORBIDDEN NODE -> Blocked!)
    └── Table: "users"

1. Visits Select object -> Is it in _FORBIDDEN_NODES? False.
2. Visits Star object -> Is it in _FORBIDDEN_NODES? False.
3. Visits Table ("orders") -> Is it in _FORBIDDEN_NODES? False.
4. Visits Column ("status") -> Is it in _FORBIDDEN_NODES? False.
5. Visits Literal ('DELETE') -> Is it in _FORBIDDEN_NODES? False. (Even though the text inside it says 'DELETE', its Python class is exp.Literal, not exp.Delete).

"""


# ----------------------------------------------------------------------------
# layer 2: the database itself refuses to write
# ----------------------------------------------------------------------------

def _authorizer(action, arg1, arg2, db_name, source):
    """SQLite calls this before every operation in a statement. Only reading is allowed."""
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION:
        if (arg2 or '').lower() in _FORBIDDEN_FUNCTIONS:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY          # INSERT, UPDATE, DELETE, CREATE, DROP, ATTACH, PRAGMA, ...


# How long one tool call may spend inside the database. Raise it (env var) if a slow but
# legitimate gold query gets cut off during evaluation.
DEFAULT_TIMEOUT_S = float(os.environ.get('SQL_TIMEOUT_S', '10'))


def connect_read_only(db_path: str, timeout_s: float = None) -> sqlite3.Connection:
    """
    Open an existing SQLite file so that writing is impossible, and so that everything run on
    this connection is aborted (sqlite3.OperationalError: interrupted) once `timeout_s` seconds
    have passed since the connection was opened.
    """
    timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s

    uri = Path(db_path).resolve().as_uri() + '?mode=ro'     # mode=ro: the file is opened read-only
    conn = sqlite3.connect(uri, uri=True)
    conn.text_factory = lambda b: b.decode(errors='ignore')
    conn.execute('PRAGMA query_only = ON')                  # second switch, same effect

    deadline = time.monotonic() + timeout_s
    # SQLite calls this every 10,000 internal steps; returning 1 aborts the running query.
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)

    conn.set_authorizer(_authorizer)                        # must come last: it blocks PRAGMA
    return conn


# ----------------------------------------------------------------------------
# layer 3: a db_id may only point inside the databases folder
# ----------------------------------------------------------------------------

"""
• Normal input: "sales" resolves to /app/data/databases/sales/sales.sqlite.
• Malicious input: An attacker passes db_id = "../../etc/secret". Without validation, joining the paths would navigate up and out of your database folder
 to access files you never intended to expose.

No .. tricks, no symlinks, no funny business: the resolved path must be a file inside the databases folder. Otherwise we refuse it.
"""

def resolve_db_path(db_base_path: str, db_id: str) -> str:
    """Return <db_base_path>/<db_id>/<db_id>.sqlite, refusing anything that escapes db_base_path."""
    base = Path(db_base_path).resolve()
    candidate = (base / db_id / f'{db_id}.sqlite').resolve()

    if base not in candidate.parents:
        raise UnsafeSQLError(f'Unknown database: {db_id!r}')
    if not candidate.is_file():
        raise UnsafeSQLError(f'Unknown database: {db_id!r}')
    return str(candidate)
