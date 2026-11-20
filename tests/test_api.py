"""Tests for api.py, using a fake model (no network, no API key) and a tiny database."""

import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from api import MAX_API_ROWS, Settings, create_app
from safe_sql import resolve_db_path


# ---------------------------------------------------------------------------
# fixtures: a tiny database, a stand-in for ExecuteSQL's module, a scriptable fake agent
# ---------------------------------------------------------------------------

@pytest.fixture
def base(tmp_path):
    db_dir = tmp_path / 'databases' / 'shop'
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / 'shop.sqlite')
    conn.execute('CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, blob BLOB)')
    conn.executemany('INSERT INTO items VALUES (?, ?, ?)',
                     [(i, f'item {i}', b'\x00\x01') for i in range(300)])
    conn.commit()
    conn.close()
    return str(tmp_path / 'databases')


SCHEMA = {'shop': {'schema': 'Tables: items(id, name)'}}
DEMOS = {'shop': {'title': 'Shop', 'description': 'A shop.', 'example_questions': ['How many items?']}}


class FakeAgent:
    """Stands in for run_agent. Behaviour can be changed per test."""

    def __init__(self):
        self.sql = 'SELECT id, name FROM items WHERE id < 3'
        self.delay = 0.0
        self.error = None
        self.block = None            # a threading.Event: wait for it before answering
        self.started = threading.Event()
        self.calls = 0

    def __call__(self, client, question, db_id, schema_lookup, db_base_path):
        self.calls += 1
        self.started.set()
        if self.block is not None:
            self.block.wait(timeout=10)
        time.sleep(self.delay)
        if self.error:
            raise self.error
        return {'predicted_sql': self.sql, 'sql_status': 'Successful', 'turns_used': 3,
                'hit_max_turns': False, 'hit_time_limit': False}


def make_api(base, agent, **settings):
    from runtime_tools import ExecuteSQL
    app = create_app(SCHEMA, client=None, db_base_path=base, run_agent=agent, execute_sql=ExecuteSQL,
                     settings=Settings(**settings), demos=DEMOS)
    return TestClient(app), app


ASK = {'db_id': 'shop', 'question': 'How many items are there?'}


@pytest.fixture
def agent():
    return FakeAgent()


# ---------------------------------------------------------------------------
# the basics
# ---------------------------------------------------------------------------

def test_health(base, agent):
    api, _ = make_api(base, agent)
    assert api.get('/health').json() == {'status': 'ok'}


def test_the_bare_url_serves_the_demo_page(base, agent):
    api, _ = make_api(base, agent)
    response = api.get('/')
    assert response.status_code == 200 and response.headers['content-type'].startswith('text/html')
    assert '/ask' in response.text and '/databases' in response.text
    assert 'innerHTML' not in response.text          # model text and cell values are only ever set as text


def test_databases_lists_the_demo_databases(base, agent):
    api, _ = make_api(base, agent)
    [db] = api.get('/databases').json()
    assert db['db_id'] == 'shop' and db['example_questions'] and db['schema']


def test_ask_returns_sql_and_rows(base, agent):
    api, _ = make_api(base, agent)
    body = api.post('/ask', json=ASK).json()
    assert body['sql'] == agent.sql
    assert body['status'] == 'Successful'
    assert body['rows'] == [[0, 'item 0'], [1, 'item 1'], [2, 'item 2']]
    assert body['truncated'] is False and body['stopped_early'] is False


def test_rows_are_capped_and_flagged(base, agent):
    agent.sql = 'SELECT id FROM items'                       # 300 rows
    api, _ = make_api(base, agent)
    body = api.post('/ask', json=ASK).json()
    assert len(body['rows']) == MAX_API_ROWS and body['truncated'] is True


def test_binary_values_do_not_break_the_json(base, agent):
    agent.sql = 'SELECT blob FROM items LIMIT 1'
    api, _ = make_api(base, agent)
    assert api.post('/ask', json=ASK).json()['rows'] == [['<2 bytes>']]


def test_columns_are_named_from_the_select_list(base, agent):
    agent.sql = 'SELECT id, name AS label, COUNT(*) FROM items WHERE id < 3 GROUP BY id'
    api, _ = make_api(base, agent)
    assert api.post('/ask', json=ASK).json()['columns'] == ['id', 'label', 'COUNT(*)']


def test_columns_are_left_out_when_unsure(base, agent):
    agent.sql = 'SELECT * FROM items WHERE id < 2'           # a star hides the names
    api, _ = make_api(base, agent)
    assert api.post('/ask', json=ASK).json()['columns'] is None


def test_the_trace_is_returned_and_bounded(base, agent):
    huge = 'x' * 5000
    original = agent.__call__

    def with_trace(*args):
        run = original(*args)
        run['trace'] = [
            {'turn': 0, 'thought': None, 'tool': 'search_column', 'args': {'query': ['singer']},
             'observation': 'singer.Name ...', 'sql_status': None},
            {'turn': 1, 'thought': 'now run it', 'tool': 'ExecuteSQL', 'args': {'sql': huge},
             'observation': huge, 'sql_status': 'Successful'},
        ]
        return run

    api, _ = make_api(base, with_trace)
    trace = api.post('/ask', json=ASK).json()['trace']
    assert [s['tool'] for s in trace] == ['search_column', 'ExecuteSQL']
    assert trace[1]['thought'] == 'now run it'
    assert len(trace[1]['observation']) < 700 and len(trace[1]['args']['sql']) < 700


def test_a_run_without_a_trace_still_answers(base, agent):
    api, _ = make_api(base, agent)                           # FakeAgent returns no trace
    assert api.post('/ask', json=ASK).json()['trace'] == []


def test_a_bad_query_from_the_agent_is_reported_not_crashed(base, agent):
    agent.sql = 'DELETE FROM items'                          # would be blocked by safe_sql
    api, _ = make_api(base, agent)
    body = api.post('/ask', json=ASK).json()
    assert body['status'] == 'Unsuccessful' and 'Blocked' in body['message'] and body['rows'] == []


# ---------------------------------------------------------------------------
# bad requests never reach the agent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('payload, code', [
    ({'db_id': 'nope', 'question': 'How many?'}, 404),
    ({'db_id': '../shop', 'question': 'How many?'}, 404),
    ({'db_id': 'shop', 'question': ''}, 422),
    ({'db_id': 'shop', 'question': '   '}, 422),
    ({'db_id': 'shop', 'question': 'x' * 301}, 422),
    ({'db_id': 'shop'}, 422),
    ({}, 422),
])
def test_invalid_requests_are_rejected(base, agent, payload, code):
    api, _ = make_api(base, agent)
    assert api.post('/ask', json=payload).status_code == code
    assert agent.calls == 0


def test_startup_fails_fast_if_a_demo_database_is_missing(base, agent):
    from runtime_tools import ExecuteSQL
    with pytest.raises(Exception):
        create_app(SCHEMA, None, base, agent, ExecuteSQL,
                   demos={'ghost': {'title': '', 'description': '', 'example_questions': []}})


# ---------------------------------------------------------------------------
# daily limits
# ---------------------------------------------------------------------------

def test_visitor_gets_five_questions_a_day(base, agent):
    api, _ = make_api(base, agent, per_client_daily=5)
    assert [api.post('/ask', json=ASK).status_code for _ in range(6)] == [200] * 5 + [429]
    assert agent.calls == 5                                   # the sixth never reached the model


def test_visitors_are_counted_separately_behind_a_proxy(base, agent):
    api, _ = make_api(base, agent, per_client_daily=1, trusted_proxy_hops=1)
    assert api.post('/ask', json=ASK, headers={'X-Forwarded-For': '1.1.1.1'}).status_code == 200
    assert api.post('/ask', json=ASK, headers={'X-Forwarded-For': '1.1.1.1'}).status_code == 429
    assert api.post('/ask', json=ASK, headers={'X-Forwarded-For': '2.2.2.2'}).status_code == 200


def test_forged_forwarded_header_does_not_dodge_the_limit(base, agent):
    api, _ = make_api(base, agent, per_client_daily=1, trusted_proxy_hops=0)
    assert api.post('/ask', json=ASK, headers={'X-Forwarded-For': '1.1.1.1'}).status_code == 200
    assert api.post('/ask', json=ASK, headers={'X-Forwarded-For': '2.2.2.2'}).status_code == 429


def test_global_daily_cap(base, agent):
    api, _ = make_api(base, agent, global_daily=2, per_client_daily=99, trusted_proxy_hops=1)
    codes = [api.post('/ask', json=ASK, headers={'X-Forwarded-For': f'9.9.9.{i}'}).status_code
             for i in range(4)]
    assert codes == [200, 200, 429, 429]


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------

def test_requests_run_at_the_same_time_not_one_after_another(base, agent):
    agent.delay = 0.6
    api, _ = make_api(base, agent, max_concurrent=4, per_client_daily=99)
    results = []

    def go():
        results.append(api.post('/ask', json=ASK).status_code)

    start = time.monotonic()
    threads = [threading.Thread(target=go) for _ in range(3)]
    for t in threads: t.start()
    for t in threads: t.join()
    elapsed = time.monotonic() - start

    assert results == [200, 200, 200]
    assert elapsed < 1.3          # one after another would take 3 x 0.6 = 1.8 seconds


def test_busy_service_says_503_and_refunds_the_question(base, agent):
    agent.block = threading.Event()
    api, app = make_api(base, agent, max_concurrent=1, slot_wait_s=0.2, per_client_daily=5)

    first = threading.Thread(target=lambda: api.post('/ask', json=ASK))
    first.start()
    assert agent.started.wait(timeout=5)                     # the first request holds the only slot

    second = api.post('/ask', json=ASK)
    assert second.status_code == 503 and second.headers['Retry-After']
    assert app.state.limiter.used('testclient') == 1         # only the first request is counted

    agent.block.set()
    first.join()


# ---------------------------------------------------------------------------
# failures reveal nothing
# ---------------------------------------------------------------------------

def test_model_failure_is_a_clean_502_without_secrets_and_is_refunded(base, agent):
    agent.error = RuntimeError('401 Unauthorized, key sk-SECRET-123 invalid')
    api, app = make_api(base, agent)
    response = api.post('/ask', json=ASK)
    assert response.status_code == 502
    assert 'sk-SECRET' not in response.text and 'RuntimeError' not in response.text
    assert app.state.limiter.used('testclient') == 0
