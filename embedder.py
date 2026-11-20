"""
embedder.py  --  the one place the embedding model is loaded and used.

Why this file exists
    Loading the model takes seconds and about 100 MB of memory. It must happen ONCE per process,
    never once per request. A server process stays alive between requests, so a model kept in a
    module-level variable is loaded one time and then reused by every request.

Three rules it follows
    1. Lazy:       importing this file loads nothing (not even the heavy sentence_transformers
                   library). The model loads on first use, or earlier if warm_up() is called.
    2. Thread-safe: if three requests arrive while the model is still loading, ONE loads it and
                   the other two wait for it. Without the lock all three would load their own
                   copy (three times the memory, three times the wait).
    3. Serialized: only one thread encodes at a time. Encoding a short phrase is quick, and the
                   threads would share the same CPU cores anyway, so this costs nothing, and it
                   avoids relying on the library being safe for concurrent use.
"""

import time
import logging
import threading

logger = logging.getLogger('sql_agent_api')

# Must be the same model preprocessing.py used to build the stored column embeddings, otherwise
# query vectors and stored vectors live in different spaces and the search returns nonsense.
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

_model = None
_load_lock = threading.Lock()
_encode_lock = threading.Lock()


def get_model():
    """Return the model, loading it if this is the first call in this process."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:      # a thread that waited on the lock finds the model already loaded
                from sentence_transformers import SentenceTransformer   # heavy import, so done lazily
                started = time.monotonic()
                _model = SentenceTransformer(EMBEDDING_MODEL)
                logger.info('embedding model loaded in %.1fs', time.monotonic() - started)
    return _model


def encode(text):
    """Embed one piece of text (returns a numpy vector)."""
    model = get_model()
    with _encode_lock:
        return model.encode(text)


def warm_up():
    """Load the model now and run one encode, so the first visitor does not pay for either."""
    encode('warm up')

"""

Why the First if Exists (Speed)

• Locks add overhead: Acquiring a lock forces threads to pause and check in with the operating system.
• 99.9% of requests skip the lock: Once the model is loaded, every future request hits that first if _model is None:, sees that _model already exists, and skips the lock completely.

2. Why the Second if Exists (Safety)

Imagine your server just booted up (_model is None), and User A and User B send requests at the exact same millisecond:
1. Both pass the 1st check: Both threads see _model is None at the same time and rush toward with _load_lock:.
2. User A wins the lock: User A enters the locked block, sees _model is None on the 2nd check, and starts loading the model (which takes ~3 seconds).
3. User B waits at the door: User B is blocked at with _load_lock: until User A finishes.
4. User B enters after 3 seconds: User A finishes loading _model and releases the lock. User B now steps inside.
5. The 2nd check saves the day: User B immediately hits the second if _model is None:. Because User A already loaded it, this is now False, so User B skips loading a duplicate copy and moves on.
"""