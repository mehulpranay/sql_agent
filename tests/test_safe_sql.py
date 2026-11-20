"""Tests for safe_sql.py. Run with:  pytest -q"""

import sqlite3

import pytest

from safe_sql import (UnsafeSQLError, check_read_only, connect_read_only, resolve_db_path)


# ---------------------------------------------------------------------------
# a tiny database in the same layout as Spider:  <base>/<db_id>/<db_id>.sqlite
# ---------------------------------------------------------------------------

@pytest.fixture
def base(tmp_path):
    db_dir = tmp_path / 'databases' / 'shop'
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / 'shop.sqlite')
    conn.execute('CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT, last_update TEXT)')
    conn.executemany('INSERT INTO items VALUES (?, ?, ?)',
                     [(1, 'pen', '2020'), (2, 'Update kit', '2021')])
    conn.commit()
    conn.close()
    (tmp_path / 'secret.sqlite').write_bytes(b'not for you')   # a file outside the folder
    return str(tmp_path / 'databases')


# ---------------------------------------------------------------------------
# layer 1: the SQL check
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sql', [
    'SELECT * FROM items',
    'select count(*) from items where id != 3',
    'SELECT name FROM items ORDER BY id DESC LIMIT 1',
    'SELECT name FROM items UNION SELECT name FROM items',
    'SELECT id FROM items INTERSECT SELECT id FROM items',
    'SELECT id FROM items EXCEPT SELECT id FROM items WHERE id = 2',
    'WITH x AS (SELECT id FROM items) SELECT * FROM x',
    'SELECT * FROM items WHERE id IN (SELECT id FROM items WHERE name = "pen")',
    'SELECT * FROM items;',                                    # one trailing semicolon is fine
    # words that look dangerous but are only data or names:
    "SELECT last_update FROM items WHERE name = 'Update kit'",
    "SELECT * FROM items WHERE name LIKE '%DROP TABLE%'",
])
def test_plain_reads_are_allowed(sql):
    check_read_only(sql)


@pytest.mark.parametrize('sql', [
    "UPDATE items SET name = 'x'",
    'DELETE FROM items',
    "INSERT INTO items VALUES (3, 'a', 'b')",
    'DROP TABLE items',
    'CREATE TABLE t (a)',
    'ALTER TABLE items ADD COLUMN c',
    "ATTACH DATABASE 'other.db' AS o",
    'PRAGMA writable_schema = 1',
    "VACUUM INTO 'copy.db'",
    'SELECT 1; DROP TABLE items',                              # a second statement hiding behind ;
    'SELECT * FROM items; DELETE FROM items;',
    "SELECT load_extension('evil')",
    "SELECT readfile('/etc/passwd')",
    '',
    '   ',
    'SELEC * FRM items',                                       # cannot be parsed
])
def test_everything_else_is_blocked(sql):
    with pytest.raises(UnsafeSQLError):
        check_read_only(sql)


# ---------------------------------------------------------------------------
# layer 2: the database refuses writes by itself.
# Here we skip layer 1 on purpose, to show the connection is safe on its own.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sql', [
    "UPDATE items SET name = 'x'",
    'DELETE FROM items',
    "INSERT INTO items VALUES (9, 'a', 'b')",
    'DROP TABLE items',
    'CREATE TABLE t (a)',
    "ATTACH DATABASE ':memory:' AS o",
    'PRAGMA query_only = OFF',
])
def test_read_only_connection_refuses_writes(base, sql):
    conn = connect_read_only(resolve_db_path(base, 'shop'))
    with pytest.raises(sqlite3.Error):
        conn.execute(sql)
    conn.close()


def test_read_only_connection_still_reads(base):
    conn = connect_read_only(resolve_db_path(base, 'shop'))
    assert conn.execute('SELECT count(*) FROM items').fetchone() == (2,)
    conn.close()


def test_data_is_unchanged_after_attacks(base):
    conn = connect_read_only(resolve_db_path(base, 'shop'))
    for sql in ('DELETE FROM items', 'DROP TABLE items'):
        with pytest.raises(sqlite3.Error):
            conn.execute(sql)
    conn.close()
    check = sqlite3.connect(resolve_db_path(base, 'shop'))
    assert check.execute('SELECT count(*) FROM items').fetchone() == (2,)
    check.close()


# ---------------------------------------------------------------------------
# layer 3: db_id cannot escape the databases folder
# ---------------------------------------------------------------------------

def test_real_db_id_resolves(base):
    assert resolve_db_path(base, 'shop').endswith('shop.sqlite')


@pytest.mark.parametrize('db_id', ['../secret', '../../etc/passwd', '/etc/passwd', 'nope', '', '..'])
def test_bad_db_ids_are_refused(base, db_id):
    with pytest.raises(UnsafeSQLError):
        resolve_db_path(base, db_id)
