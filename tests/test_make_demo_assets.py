"""Tests for make_demo_assets.py: the folder of files the container ships with."""

import pickle
from importlib import metadata

import numpy as np
import pytest

from demo_dbs import DEMO_DATABASES
from make_demo_assets import build_assets
from tests.conftest import make_lookup_entry


def test_only_the_demo_databases_are_included(fake_spider, tmp_path):
    pkl, db_dir = fake_spider
    out = build_assets(pkl, db_dir, tmp_path / 'demo_assets')

    with open(out / 'spider_schema_lookup.pkl', 'rb') as f:
        trimmed = pickle.load(f)
    assert set(trimmed) == set(DEMO_DATABASES)                      # 'some_other_db' left out
    for db_id in DEMO_DATABASES:
        assert (out / 'databases' / db_id / f'{db_id}.sqlite').is_file()
    assert not (out / 'databases' / 'some_other_db').exists()
    assert (out / 'ATTRIBUTION.md').is_file()


def test_versions_that_made_the_pickle_are_pinned(fake_spider, tmp_path):
    pkl, db_dir = fake_spider
    out = build_assets(pkl, db_dir, tmp_path / 'demo_assets')
    constraints = (out / 'constraints.txt').read_text().splitlines()
    assert f'numpy=={metadata.version("numpy")}' in constraints
    assert f'networkx=={metadata.version("networkx")}' in constraints


def test_old_files_do_not_linger(fake_spider, tmp_path):
    pkl, db_dir = fake_spider
    out = tmp_path / 'demo_assets'
    (out / 'databases' / 'removed_db').mkdir(parents=True)
    (out / 'databases' / 'removed_db' / 'removed_db.sqlite').write_bytes(b'old')
    build_assets(pkl, db_dir, out)
    assert not (out / 'databases' / 'removed_db').exists()


def test_a_demo_database_missing_from_the_lookup_is_reported(fake_spider, tmp_path):
    pkl, db_dir = fake_spider
    with open(pkl, 'rb') as f:
        lookup = pickle.load(f)
    del lookup[next(iter(DEMO_DATABASES))]
    with open(pkl, 'wb') as f:
        pickle.dump(lookup, f)
    with pytest.raises(SystemExit, match='is not in'):
        build_assets(pkl, db_dir, tmp_path / 'demo_assets')


def test_a_missing_sqlite_file_is_reported(fake_spider, tmp_path):
    pkl, db_dir = fake_spider
    first = next(iter(DEMO_DATABASES))
    (db_dir / first / f'{first}.sqlite').unlink()
    with pytest.raises(SystemExit, match='not found'):
        build_assets(pkl, db_dir, tmp_path / 'demo_assets')


def test_embeddings_that_do_not_match_the_columns_are_reported(fake_spider, tmp_path):
    pkl, db_dir = fake_spider
    first = next(iter(DEMO_DATABASES))
    with open(pkl, 'rb') as f:
        lookup = pickle.load(f)
    lookup[first]['embeddings'] = np.zeros((5, 384))                # 3 columns but 5 rows
    with open(pkl, 'wb') as f:
        pickle.dump(lookup, f)
    with pytest.raises(SystemExit, match='embeddings have shape'):
        build_assets(pkl, db_dir, tmp_path / 'demo_assets')


def test_a_pickle_that_needs_pandas_is_caught_before_it_breaks_the_container(fake_spider, tmp_path):
    pandas = pytest.importorskip('pandas')
    pkl, db_dir = fake_spider
    first = next(iter(DEMO_DATABASES))
    with open(pkl, 'rb') as f:
        lookup = pickle.load(f)
    lookup[first]['stats'] = pandas.DataFrame({'a': [1]})            # a stray pandas object
    with open(pkl, 'wb') as f:
        pickle.dump(lookup, f)
    with pytest.raises(SystemExit, match='service image does not install'):
        build_assets(pkl, db_dir, tmp_path / 'demo_assets')
