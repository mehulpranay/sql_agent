"""
Test the real start-up path (api.build_app) the way the container runs it: load the schema file from
demo_assets, load the model, start serving. Only the model and the DeepSeek client are stand-ins.
"""

import pytest
from fastapi.testclient import TestClient

import agent_demo
import embedder
import runtime_tools
from demo_dbs import DEMO_DATABASES
from make_demo_assets import build_assets
from api import build_app


@pytest.fixture
def app_from_assets(fake_spider, tmp_path, monkeypatch):
    pkl, db_dir = fake_spider
    assets = build_assets(pkl, db_dir, tmp_path / 'demo_assets')

    # point the service at the trimmed assets, exactly like the Dockerfile's environment variables
    monkeypatch.setattr(runtime_tools, 'DB_BASE_PATH', str(assets / 'databases'))
    monkeypatch.setattr(runtime_tools, 'SCHEMA_PKL_PATH', str(assets / 'spider_schema_lookup.pkl'))
    monkeypatch.setattr(agent_demo, 'make_client', lambda: None)       # no real DeepSeek client
    monkeypatch.setattr(embedder, '_model', None)                      # start with the model not loaded
    return build_app()


def test_the_service_starts_from_the_shipped_assets(app_from_assets):
    api = TestClient(app_from_assets)
    assert api.get('/health').status_code == 200
    ids = [db['db_id'] for db in api.get('/databases').json()]
    assert ids == list(DEMO_DATABASES)


def test_the_model_is_already_loaded_when_the_first_visitor_arrives(app_from_assets):
    assert embedder._model is not None


def test_a_model_service_failure_is_a_clean_error(app_from_assets, monkeypatch):
    def broken_agent(*args, **kwargs):
        raise ConnectionError('could not reach the model API')

    # build_app already captured run_agent, so rebuild with the broken one
    monkeypatch.setattr(agent_demo, 'run_agent', broken_agent)
    api = TestClient(build_app())
    first = next(iter(DEMO_DATABASES))
    response = api.post('/ask', json={'db_id': first, 'question': 'How many rows are there?'})
    assert response.status_code == 502 and 'ConnectionError' not in response.text
