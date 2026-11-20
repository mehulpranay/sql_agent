"""Shared test helper: a miniature "Spider download" with the same layout as the real one."""

import pickle
import sqlite3

import networkx as nx
import numpy as np
import pytest

from demo_dbs import DEMO_DATABASES


def make_lookup_entry(db_id, n_cols=3):
    col_dicts = [{'table_name': 't', 'column_name': f'c{i}', 'column_type': 'text',
                  'statistics': 'N/A', 'foreign_key': None} for i in range(n_cols)]
    return {'db_id': db_id, 'schema': f'Tables: t(c0, c1, c2) in {db_id}', 'col_dicts': col_dicts,
            'embeddings': np.zeros((n_cols, 384)), 'graph': nx.Graph()}


@pytest.fixture
def fake_spider(tmp_path):
    """Returns (path of the full schema pickle, path of the database folder)."""
    db_dir = tmp_path / 'spider' / 'database'
    lookup = {}
    for db_id in [*DEMO_DATABASES, 'some_other_db']:
        (db_dir / db_id).mkdir(parents=True)
        conn = sqlite3.connect(db_dir / db_id / f'{db_id}.sqlite')
        conn.execute('CREATE TABLE t (c0 TEXT, c1 TEXT, c2 TEXT)')
        conn.commit()
        conn.close()
        lookup[db_id] = make_lookup_entry(db_id)

    pkl = tmp_path / 'full_lookup.pkl'
    with open(pkl, 'wb') as f:
        pickle.dump(lookup, f)
    return pkl, db_dir
