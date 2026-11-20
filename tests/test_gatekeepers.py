"""Tests for limits.py: the daily limiter, the concurrency gate and the visitor identity."""

import threading
import time
from datetime import date, timedelta

import pytest

from limits import Busy, ConcurrencyGate, DailyLimiter, client_id_from


# ---------------------------------------------------------------------------
# DailyLimiter
# ---------------------------------------------------------------------------

def test_visitor_limit():
    limiter = DailyLimiter(per_client=2, global_limit=100)
    assert limiter.reserve('a') is None
    assert limiter.reserve('a') is None
    assert limiter.reserve('a') == 'client'          # third question is refused
    assert limiter.reserve('b') is None              # another visitor is unaffected


def test_global_limit_beats_visitor_limit():
    limiter = DailyLimiter(per_client=5, global_limit=2)
    assert limiter.reserve('a') is None
    assert limiter.reserve('b') is None
    assert limiter.reserve('c') == 'global'


def test_refused_questions_are_not_counted():
    limiter = DailyLimiter(per_client=1, global_limit=100)
    limiter.reserve('a')
    for _ in range(50):
        assert limiter.reserve('a') == 'client'
    assert limiter.used('a') == 1


def test_refund_gives_a_question_back():
    limiter = DailyLimiter(per_client=1, global_limit=1)
    limiter.reserve('a')
    limiter.refund('a')
    assert limiter.reserve('a') is None              # allowed again, per-visitor and global


def test_counts_reset_on_a_new_day():
    today = [date(2026, 11, 9)]
    limiter = DailyLimiter(per_client=1, global_limit=1, today=lambda: today[0])
    assert limiter.reserve('a') is None
    assert limiter.reserve('a') is not None
    today[0] += timedelta(days=1)
    assert limiter.reserve('a') is None


def test_many_threads_cannot_exceed_the_limit():
    limiter = DailyLimiter(per_client=10, global_limit=10_000)
    allowed = []

    def attempt():
        if limiter.reserve('same visitor') is None:
            allowed.append(1)

    threads = [threading.Thread(target=attempt) for _ in range(200)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(allowed) == 10


# ---------------------------------------------------------------------------
# ConcurrencyGate
# ---------------------------------------------------------------------------

def test_gate_refuses_when_full_and_recovers():
    gate = ConcurrencyGate(max_concurrent=1, wait_s=0.05)
    with gate.slot():
        with pytest.raises(Busy):
            with gate.slot():
                pass
    with gate.slot():                                # the slot was released
        pass


def test_gate_releases_the_slot_after_an_error():
    gate = ConcurrencyGate(max_concurrent=1, wait_s=0.05)
    with pytest.raises(RuntimeError):
        with gate.slot():
            raise RuntimeError('agent crashed')
    with gate.slot():
        pass


def test_waiting_request_gets_the_slot_when_it_frees_up():
    gate = ConcurrencyGate(max_concurrent=1, wait_s=2)
    got_it = []

    def second():
        with gate.slot():
            got_it.append(True)

    with gate.slot():
        t = threading.Thread(target=second)
        t.start()
        time.sleep(0.2)                              # the second request is waiting
    t.join()
    assert got_it == [True]


# ---------------------------------------------------------------------------
# who is "the visitor"?
# ---------------------------------------------------------------------------

def test_direct_connection_ignores_forwarded_header():
    # hops = 0: X-Forwarded-For is attacker-controlled, so it must not decide anything
    assert client_id_from('1.2.3.4', '9.9.9.9', 0) == '1.2.3.4'


def test_behind_one_proxy_uses_the_entry_the_proxy_added():
    # a visitor can write the left side; the proxy appends the real address on the right
    assert client_id_from('10.0.0.1', 'fake-1, fake-2, 5.6.7.8', 1) == '5.6.7.8'


def test_behind_two_proxies():
    assert client_id_from('10.0.0.1', 'fake, 5.6.7.8, 10.1.1.1', 2) == '5.6.7.8'


def test_falls_back_when_header_is_missing_or_too_short():
    assert client_id_from('1.2.3.4', '', 1) == '1.2.3.4'
    assert client_id_from('1.2.3.4', '5.6.7.8', 2) == '1.2.3.4'
