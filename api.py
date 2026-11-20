"""
api.py  --  the web API in front of the SQL agent.

    GET  /health      is the service up? (hosting platforms poll this)
    GET  /databases   the demo databases, with a description and example questions
    POST /ask         {"db_id": "...", "question": "..."}  ->  the SQL, the rows and how the run went
    GET  /            the demo page (web/index.html): pick a database, ask, see the SQL and the steps
    GET  /docs        automatic interactive page: you can try the API from the browser

What this file is responsible for, in the order a request meets it:
    1. is the request well-formed, and is the database one we offer?        (validation)
    2. has this visitor, or the whole demo, used up today's questions?      (DailyLimiter)
    3. is there a free slot to run the agent right now?                     (ConcurrencyGate)
    4. run the agent, then fetch a few rows of its final query              (agent + ExecuteSQL)
    5. turn failures into clean errors that reveal nothing internal         (error handling)

Run it with ONE worker (the counters live in memory):
    uvicorn api:build_app --factory --host 0.0.0.0 --port 8000
"""

import os
import time
import pickle
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from demo_dbs import DEMO_DATABASES
from limits import Busy, ConcurrencyGate, DailyLimiter, client_id_from
from safe_sql import resolve_db_path

logger = logging.getLogger('sql_agent_api')

MAX_API_ROWS = 100          # rows returned to the visitor; more are cut and flagged as truncated
MAX_TRACE_TEXT = 600        # characters of one step's thought / observation shown to the visitor
PAGE_PATH = Path(__file__).parent / 'web' / 'index.html'


@dataclass
class Settings:
    per_client_daily: int = 5       # questions one visitor may ask per day
    global_daily: int = 200         # questions the whole demo answers per day (your credit budget)
    max_concurrent: int = 4         # agent runs at the same time
    slot_wait_s: float = 5.0        # how long a request waits for a free slot before "busy"
    trusted_proxy_hops: int = 0     # 0 = exposed directly; 1 = behind one platform proxy (see limits.py)

    @classmethod
    def from_env(cls):
        return cls(
            per_client_daily=int(os.environ.get('DAILY_PER_CLIENT', '5')),
            global_daily=int(os.environ.get('DAILY_GLOBAL', '200')),
            max_concurrent=int(os.environ.get('MAX_CONCURRENT', '4')),
            slot_wait_s=float(os.environ.get('SLOT_WAIT_S', '5')),
            trusted_proxy_hops=int(os.environ.get('TRUSTED_PROXY_HOPS', '0')),
        )


# ----------------------------------------------------------------------------
# what comes in and what goes out
# ----------------------------------------------------------------------------

class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    db_id: str = Field(max_length=64)
    question: str = Field(min_length=3, max_length=300)     # a question, not an essay


class TraceStep(BaseModel):
    """One turn of the agent: what it said, which tool it called, and what came back."""
    turn: int
    thought: Optional[str]
    tool: str
    args: Optional[dict]
    observation: Optional[str]


class AskResponse(BaseModel):
    db_id: str
    question: str
    sql: Optional[str]              # the agent's final query (None if it never produced one)
    status: str                     # 'Successful', 'Unsuccessful' or 'SQL never generated'
    message: Optional[str]          # the error text when status is not 'Successful'
    columns: Optional[List[str]]    # best-effort column names for the rows (None when unsure)
    rows: List[List[Any]]           # at most MAX_API_ROWS rows
    truncated: bool                 # True when the query had more rows than we return
    turns_used: int
    stopped_early: bool             # the agent hit its turn or time limit instead of finishing
    seconds: float
    trace: List[TraceStep]          # the agent's steps, for the "how did it get here" view


def _json_safe(value):
    """SQLite can hold raw bytes (BLOBs); JSON cannot, so describe them instead."""
    return f'<{len(value)} bytes>' if isinstance(value, (bytes, bytearray)) else value


def _short(text, limit=MAX_TRACE_TEXT):
    if text is None:
        return None
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f' ...({len(text) - limit} more characters)'


def _public_trace(trace):
    """The agent's trace, with every text bounded so one huge cell cannot bloat the response."""
    steps = []
    for step in trace or []:
        args = step.get('args')
        steps.append(TraceStep(
            turn=int(step.get('turn', len(steps))),
            thought=_short(step.get('thought')),
            tool=str(step.get('tool', '')),
            args={k: _short(v) if isinstance(v, str) else v for k, v in args.items()}
                 if isinstance(args, dict) else None,
            observation=_short(step.get('observation')),
        ))
    return steps


def _column_names(sql, width):
    """
    Names for the result columns, read from the SELECT list of the final query.
    Best effort: with `SELECT *`, or when the count does not match the rows, return None
    and the page shows the rows without a header.
    """
    try:
        import sqlglot
        from sqlglot import exp
        tree = sqlglot.parse_one(sql, read='sqlite')
        selects = tree.selects
    except Exception:
        return None
    if not selects or len(selects) != width or any(isinstance(s.unalias(), exp.Star) for s in selects):
        return None
    names = []
    for s in selects:
        # an alias or a plain column keeps its name; anything else (COUNT(*), a calculation) is shown as written
        name = s.alias if isinstance(s, exp.Alias) else s.name if isinstance(s, exp.Column) else s.sql(dialect='sqlite')
        names.append(name if len(name) <= 40 else name[:37] + '...')
    return names


# ----------------------------------------------------------------------------
# the app
# ----------------------------------------------------------------------------

def create_app(schema_lookup, client, db_base_path, run_agent, execute_sql,
               settings: Settings = None, demos: dict = None):
    """
    Everything the app needs is passed in, so tests can hand it a fake model and a tiny database,
    and production hands it the real ones (see build_app below).
    """

    settings = settings or Settings()
    demos = DEMO_DATABASES if demos is None else demos

    # Fail at startup, not at the first visitor, if a demo database is misconfigured.
    for db_id in demos:
        if db_id not in schema_lookup:
            raise RuntimeError(f'{db_id!r} is in the demo list but not in the schema lookup.')
        resolve_db_path(db_base_path, db_id)

    limiter = DailyLimiter(settings.per_client_daily, settings.global_daily)
    gate = ConcurrencyGate(settings.max_concurrent, settings.slot_wait_s)

    app = FastAPI(title='SQL agent', description='Ask a question about a database; an agent writes the SQL.')
    app.state.limiter = limiter         # exposed so tests can look at the counters

    # The demo page is one static file, read once at startup (so a missing file fails the start,
    # not the first visitor). It is plain HTML + JavaScript that calls /databases and /ask.
    page = PAGE_PATH.read_text(encoding='utf-8')

    @app.get('/', response_class=HTMLResponse, include_in_schema=False)
    def home():
        return HTMLResponse(page)

    @app.get('/health')
    def health():
        return {'status': 'ok'}

    @app.get('/databases')
    def databases():
        return [
            {'db_id': db_id, **info, 'schema': schema_lookup[db_id]['schema']}
            for db_id, info in demos.items()
        ]

    # `def`, not `async def`, on purpose. The agent makes blocking calls (the model's HTTP API, the
    # database). In an `async def`, a blocking call freezes the server's single event loop, so every
    # other visitor would wait. A plain `def` is run by FastAPI in a pool of worker threads, so
    # slow requests overlap instead of queueing.
    @app.post('/ask', response_model=AskResponse)
    def ask(body: AskRequest, request: Request):

        # 1. is it a database we offer?
        if body.db_id not in demos:
            raise HTTPException(404, f'Unknown database. Choose one of: {", ".join(demos)}.')

        # 2. daily limits. Done BEFORE taking a slot, so a flood is turned away cheaply.
        visitor = client_id_from(
            request.client.host if request.client else '',
            request.headers.get('x-forwarded-for', ''),
            settings.trusted_proxy_hops,
        )
        refusal = limiter.reserve(visitor)
        if refusal == 'client':
            raise HTTPException(429, f'Limit reached: {settings.per_client_daily} questions per day. '
                                     f'Please come back tomorrow.')
        if refusal == 'global':
            raise HTTPException(429, 'The demo has reached its daily limit. Please try again tomorrow.')

        # 3 + 4. wait briefly for a free slot, then run the agent
        started = time.monotonic()
        try:
            with gate.slot():
                run = run_agent(client, body.question, body.db_id, schema_lookup, db_base_path)
        except Busy:
            limiter.refund(visitor)                      # the visitor got nothing, so it costs nothing
            raise HTTPException(503, 'The service is busy right now. Please try again in a moment.',
                                headers={'Retry-After': '10'})
        except Exception:
            limiter.refund(visitor)
            logger.exception('agent run failed (db_id=%s)', body.db_id)   # details stay in OUR logs
            raise HTTPException(502, 'The language model service failed. Please try again later.')

        # the final query's rows (the agent returns the SQL; we fetch a limited number of rows)
        sql, status, message = run['predicted_sql'], run['sql_status'], None
        rows, truncated, columns = [], False, None
        if sql:
            status, fetched = execute_sql(sql, body.db_id, db_base_path, max_rows=MAX_API_ROWS + 1)
            if status == 'Successful':
                truncated = len(fetched) > MAX_API_ROWS
                rows = [[_json_safe(v) for v in row] for row in fetched[:MAX_API_ROWS]]
                columns = _column_names(sql, len(rows[0])) if rows else None
            else:
                message = str(fetched)

        seconds = round(time.monotonic() - started, 1)
        logger.info('ask db=%s turns=%s status=%s seconds=%s', body.db_id, run['turns_used'], status, seconds)

        return AskResponse(
            db_id=body.db_id, question=body.question, sql=sql, status=status, message=message,
            columns=columns, rows=rows, truncated=truncated, turns_used=run['turns_used'],
            stopped_early=bool(run.get('hit_max_turns') or run.get('hit_time_limit')),
            seconds=seconds, trace=_public_trace(run.get('trace')),
        )

    return app


def build_app():
    """The real app: real model client, real schema lookup, settings from environment variables."""
    # Imported here so that importing api.py (e.g. in tests) does not load the embedding model.
    from agent_demo import make_client, run_agent
    from embedder import warm_up
    from runtime_tools import DB_BASE_PATH, SCHEMA_PKL_PATH, ExecuteSQL

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    started = time.monotonic()
    with open(SCHEMA_PKL_PATH, 'rb') as f:
        schema_lookup = pickle.load(f)
    logger.info('schema lookup loaded in %.1fs', time.monotonic() - started)

    app = create_app(schema_lookup, make_client(), DB_BASE_PATH, run_agent, ExecuteSQL,
                     Settings.from_env())

    # Load the embedding model BEFORE the server starts listening. uvicorn opens its port only
    # after this function returns, so a visitor (or the host's health check) can never reach
    # a server that is still loading, and nobody's first question pays for the model load.
    warm_up()
    return app
