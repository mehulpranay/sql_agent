"""Tests for embedder.py: the model loads once, safely, and importing code loads nothing."""

import subprocess
import sys
import threading
import time
import types

import pytest

import embedder


@pytest.fixture
def counting_model(monkeypatch):
    """Replace the real model with a stand-in that counts loads and overlapping encodes."""
    state = types.SimpleNamespace(loads=0, running=0, max_running=0, lock=threading.Lock())

    class FakeModel:
        def __init__(self, name):
            time.sleep(0.2)                         # loading takes a while, so threads really race
            with state.lock:
                state.loads += 1

        def encode(self, text):
            with state.lock:
                state.running += 1
                state.max_running = max(state.max_running, state.running)
            time.sleep(0.02)
            with state.lock:
                state.running -= 1
            return [0.0] * 384

    fake_module = types.ModuleType('sentence_transformers')
    fake_module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, 'sentence_transformers', fake_module)
    monkeypatch.setattr(embedder, '_model', None)
    return state


def run_in_threads(fn, count):
    threads = [threading.Thread(target=fn) for _ in range(count)]
    for t in threads: t.start()
    for t in threads: t.join()


def test_model_loads_once_even_when_many_requests_race_for_it(counting_model):
    run_in_threads(lambda: embedder.encode('hello'), 20)
    assert counting_model.loads == 1


def test_encodes_do_not_overlap(counting_model):
    embedder.warm_up()
    run_in_threads(lambda: embedder.encode('hello'), 10)
    assert counting_model.max_running == 1


def test_model_is_reused_not_reloaded(counting_model):
    for _ in range(5):
        embedder.encode('hello')
    assert counting_model.loads == 1


def test_warm_up_loads_the_model_before_any_request(counting_model):
    assert embedder._model is None
    embedder.warm_up()
    assert embedder._model is not None and counting_model.loads == 1


def test_a_failed_load_is_retried_not_remembered(monkeypatch):
    calls = []

    class Flaky:
        def __init__(self, name):
            calls.append(1)
            if len(calls) == 1:
                raise OSError('download failed')

        def encode(self, text):
            return [0.0]

    fake_module = types.ModuleType('sentence_transformers')
    fake_module.SentenceTransformer = Flaky
    monkeypatch.setitem(sys.modules, 'sentence_transformers', fake_module)
    monkeypatch.setattr(embedder, '_model', None)

    with pytest.raises(OSError):
        embedder.encode('x')
    assert embedder.encode('x') == [0.0]            # the second attempt works


def test_importing_the_tools_loads_nothing():
    """In a fresh interpreter: import the tools and check that no model and no heavy library loaded."""
    code = (
        "import sys; import runtime_tools, embedder; "
        "print(embedder._model is None, 'sentence_transformers' in sys.modules)"
    )
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                         cwd=str(__import__('pathlib').Path(__file__).resolve().parent.parent))
    assert out.stdout.strip() == 'True False', out.stderr
