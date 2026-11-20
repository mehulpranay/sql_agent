"""Tests for the time and size limits. Run with:  pytest -q"""

import json
import sqlite3
import time
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import agent_demo
import runtime_tools
import safe_sql
from runtime_tools import ExecuteSQL, search_value
from safe_sql import connect_read_only, resolve_db_path

BIG_TABLE_ROWS = 3000


@pytest.fixture
def base(tmp_path):
    db_dir = tmp_path / 'databases' / 'shop'
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / 'shop.sqlite')
    conn.execute('CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)')
    conn.executemany('INSERT INTO items VALUES (?, ?)',
                     [(i, f'item {i}') for i in range(BIG_TABLE_ROWS)])
    conn.commit()
    conn.close()
    return str(tmp_path / 'databases')


FOREVER = 'WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r'
CROSS_JOIN = 'SELECT count(*) FROM items a, items b, items c'      # 27 billion row combinations


# ---------------------------------------------------------------------------
# the time limit
# ---------------------------------------------------------------------------

def test_endless_query_is_stopped(base):
    conn = connect_read_only(resolve_db_path(base, 'shop'), timeout_s=0.3)
    start = time.monotonic()
    with pytest.raises(sqlite3.OperationalError, match='interrupted'):
        conn.execute(FOREVER).fetchall()
    assert time.monotonic() - start < 3
    conn.close()


def test_execute_sql_reports_a_timeout_as_an_error(base, monkeypatch):
    monkeypatch.setattr(safe_sql, 'DEFAULT_TIMEOUT_S', 0.5)
    start = time.monotonic()
    status, message = ExecuteSQL(CROSS_JOIN, 'shop', base)
    assert time.monotonic() - start < 5
    assert status == 'Unsuccessful'
    assert 'ran too long' in message                        # the agent can read this and fix its query


def test_normal_queries_are_not_affected(base):
    status, rows = ExecuteSQL('SELECT count(*) FROM items', 'shop', base)
    assert (status, rows) == ('Successful', [(BIG_TABLE_ROWS,)])


# ---------------------------------------------------------------------------
# the row limit
# ---------------------------------------------------------------------------

def test_max_rows_stops_reading_early(base):
    status, rows = ExecuteSQL('SELECT * FROM items', 'shop', base, max_rows=5)
    assert status == 'Successful' and len(rows) == 5


def test_no_max_rows_means_everything(base):          # evaluation needs complete results
    status, rows = ExecuteSQL('SELECT * FROM items', 'shop', base)
    assert len(rows) == BIG_TABLE_ROWS


# ---------------------------------------------------------------------------
# search_value uses the same safe connection
# ---------------------------------------------------------------------------

def test_search_value_still_works(base):
    col_dicts = [{'table_name': 'items', 'column_name': 'name', 'column_type': 'text'}]
    result = search_value('item 42', 'shop', col_dicts, base)
    assert {'contents': 'item 42', 'table': 'items', 'column': 'name'} in result


def test_search_value_refuses_a_bad_db_id(base):
    assert 'Unknown database' in search_value('x', '../shop', [], base)


# ---------------------------------------------------------------------------
# the size limit on what enters the model's context
# ---------------------------------------------------------------------------

def test_limit_text():
    assert agent_demo.limit_text('short') == 'short'
    cut = agent_demo.limit_text('x' * 10_000)
    assert len(cut) < agent_demo.MAX_OBSERVATION_CHARS + 100 and 'cut:' in cut


# ---------------------------------------------------------------------------
# the agent loop, driven by a scripted fake model (no network, no API key)
# ---------------------------------------------------------------------------

class FakeMessage:
    def __init__(self, tool_name, arguments):
        call = SimpleNamespace(id='call_1',
                               function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments)))
        self.content = 'Thought: scripted'
        self.tool_calls = [call]
        self._dump = {'role': 'assistant', 'content': self.content,
                      'tool_calls': [{'id': 'call_1', 'type': 'function',
                                      'function': {'name': tool_name, 'arguments': json.dumps(arguments)}}]}

    def model_dump(self, exclude_none=True):
        return self._dump


class FakeClient:
    """Answers each call with the next scripted (tool, arguments) pair."""

    def __init__(self, script):
        self.script = list(script)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        tool, arguments = self.script.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=FakeMessage(tool, arguments))])


@pytest.fixture
def schema_lookup():
    return {'shop': {
        'schema': 'Tables: items(id, name)\nForeignKeys: none',
        'col_dicts': [{'table_name': 'items', 'column_name': 'name', 'column_type': 'text',
                       'statistics': 'N/A'}],
        'embeddings': np.zeros((1, 384)),
        'graph': nx.Graph(),
    }}


def run(script, base, schema_lookup):
    return agent_demo.run_agent(FakeClient(script), 'a question', 'shop', schema_lookup, base)


def test_loop_finishes_normally(base, schema_lookup):
    result = run([('ExecuteSQL', {'sql': 'SELECT count(*) FROM items'}), ('Done', {})], base, schema_lookup)
    assert result['predicted_sql'] == 'SELECT count(*) FROM items'
    assert result['sql_status'] == 'Successful'
    assert not result['hit_time_limit'] and not result['hit_max_turns']


def test_loop_shows_the_model_that_a_write_was_blocked(base, schema_lookup):
    result = run([('ExecuteSQL', {'sql': 'DELETE FROM items'}), ('Done', {})], base, schema_lookup)
    first = result['trace'][0]
    assert first['sql_status'] == 'Unsuccessful' and 'Blocked' in first['observation']
    # and the data is still there
    assert ExecuteSQL('SELECT count(*) FROM items', 'shop', base)[1] == [(BIG_TABLE_ROWS,)]


def test_loop_caps_a_huge_result(base, schema_lookup):
    result = run([('ExecuteSQL', {'sql': 'SELECT * FROM items'}), ('Done', {})], base, schema_lookup)
    observation = result['trace'][0]['observation']
    assert 'more than 1000 rows' in observation
    assert len(observation) <= agent_demo.MAX_OBSERVATION_CHARS + 100


def test_loop_stops_at_the_time_limit(base, schema_lookup, monkeypatch):
    monkeypatch.setattr(agent_demo, 'MAX_SECONDS', -1)       # already out of time
    result = run([], base, schema_lookup)                     # the model is never even called
    assert result['hit_time_limit'] and result['turns_used'] == 0


def test_loop_stops_at_max_turns(base, schema_lookup):
    script = [('ExecuteSQL', {'sql': 'SELECT 1'})] * agent_demo.MAX_TURNS
    result = run(script, base, schema_lookup)
    assert result['hit_max_turns'] and result['turns_used'] == agent_demo.MAX_TURNS
