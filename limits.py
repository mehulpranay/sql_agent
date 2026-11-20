"""
limits.py  --  the two gatekeepers that sit in front of the agent.

    DailyLimiter       "how many questions may this visitor, and the whole demo, ask today?"
                       Protects your DeepSeek credits.
    ConcurrencyGate    "how many agent runs may be in progress at the same moment?"
                       Protects the server: CPU, memory and the model's rate limit.

Both keep their counts in this process's memory. That is correct while the service runs as ONE
process (one uvicorn worker, one container). If you ever run several copies, each would count on
its own and the limits would be multiplied; you would then move the counters to Redis or similar.
"""

import threading
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone


def _utc_today():
    return datetime.now(timezone.utc).date()


"""
Need of Threading Lock: The DailyLimiter is used in a multi-threaded environment where multiple threads can access and modify the counts of questions asked by visitors. 
Without a lock, two threads could simultaneously check and update the counts, leading to race conditions and incorrect counts.

When multiple users send requests to your server at the same time, different threads run reserve() concurrently.

• Without a lock (Race Condition): Imagine global_limit is 100 and self._total is 99. Two users arrive at the exact same time. Both threads check if self._total >= 100: 
    and see 99. Both pass the check, and both add 1, pushing your total to 101 and bypassing your limit.

• With with self._lock:: Python forces threads to line up single-file. Only one thread can enter the with self._lock: block at a time. Everyone else waits until 
    that thread finishes checking and updating the numbers

Example Scenario:
1. Visitor A and Visitor B arrive at the same instant and get assigned to Thread 1 and Thread 2.
2. Both threads call limiter.reserve(client_id). with self._lock: makes Thread 2 wait for a few nanoseconds while Thread 1 updates the counter.

"""


class DailyLimiter:
    """Counts questions per visitor and in total, per UTC day."""

    def __init__(self, per_client: int, global_limit: int, today=_utc_today):
        self.per_client = per_client
        self.global_limit = global_limit
        self._today = today
        self._lock = threading.Lock()          # requests run in several threads at once
        self._day = None
        self._counts = defaultdict(int)
        self._total = 0

    def _roll_over(self):
        today = self._today()
        if today != self._day:                 # a new day: everyone starts again from zero
            self._day, self._counts, self._total = today, defaultdict(int), 0

    def reserve(self, client_id: str):
        """
        Try to use up one question. Returns None if allowed, otherwise 'global' or 'client'
        (the reason). Nothing is counted when the answer is "not allowed".
        """
        with self._lock:
            self._roll_over()
            if self._total >= self.global_limit:
                return 'global'
            if self._counts[client_id] >= self.per_client:
                return 'client'
            self._counts[client_id] += 1
            self._total += 1
            return None

    def refund(self, client_id: str):
        """Give a question back, e.g. when we turned the visitor away or the model service failed."""
        with self._lock:
            self._roll_over()
            if self._counts[client_id] > 0:
                self._counts[client_id] -= 1
                self._total -= 1

    def used(self, client_id: str) -> int:
        with self._lock:
            self._roll_over()
            return self._counts[client_id]


class Busy(Exception):
    """All agent slots are taken and none became free in time."""


class ConcurrencyGate:
    """At most `max_concurrent` agent runs at once; others wait up to `wait_s`, then are refused."""

    def __init__(self, max_concurrent: int, wait_s: float):
        self._slots = threading.BoundedSemaphore(max_concurrent)
        self._wait_s = wait_s

    @contextmanager
    def slot(self):
        if not self._slots.acquire(timeout=self._wait_s):
            raise Busy()
        try:
            yield
        finally:
            self._slots.release()


def client_id_from(client_host: str, forwarded_for: str, trusted_proxy_hops: int) -> str:
    """
    Decide who "the visitor" is.

    Directly exposed (hops = 0): use the address of the connection. Never trust X-Forwarded-For
    then, because anyone can send that header with any value they like.

    Behind a platform proxy (hops = 1, or 2 if there are two proxies): each proxy ADDS the
    address it saw to the END of X-Forwarded-For. So the address added by the proxy nearest to
    us, the one we can trust, is `hops` entries from the right. Entries further left could have
    been written by the visitor.
    """
    if trusted_proxy_hops > 0 and forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(',') if p.strip()]
        if len(parts) >= trusted_proxy_hops:
            return parts[-trusted_proxy_hops]
    return client_host or 'unknown'
