"""
pytest setup, shared by every test.

- Having this file in the repo root also makes pytest add the root to the import path, so
  `import safe_sql` works when you run plain `pytest`.
- The embedding model is replaced by a do-nothing stand-in: tests must not download or load a
  real model, and none of them needs real embeddings.
"""

import sys
import types

_stub = types.ModuleType('sentence_transformers')


class _FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text):
        import numpy as np
        return np.zeros(384)


_stub.SentenceTransformer = _FakeSentenceTransformer
sys.modules['sentence_transformers'] = _stub
