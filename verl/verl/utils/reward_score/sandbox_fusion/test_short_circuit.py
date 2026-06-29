# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for binary-verdict short-circuit cancellation in check_correctness.

A correct solution must still run *every* test case (binary all-must-pass), but
a wrong one should stop at the first failure instead of burning a sandbox slot
on every remaining case.
"""

import threading
import time

from verl.utils.reward_score.sandbox_fusion import utils
from verl.utils.reward_score.sandbox_fusion import compute_score

N = 20


def _fake_call_factory(verdict, delay=0.0):
    """Build a stand-in for call_sandbox_api that records invocations.

    verdict controls the simulated run outcome. ``delay`` models the fact that a
    real sandbox call takes nonzero time (seconds), so the short-circuit signal
    is observed before the queue can drain -- without it the instant fake lets
    workers race ahead of the main loop, which never happens in production.
    """
    calls = []
    lock = threading.Lock()

    def fake_call(sandbox_fusion_url, code, stdin, compile_timeout, run_timeout, memory_limit_mb, language="python", retry_on_timeout=True):
        with lock:
            calls.append(stdin)
        if delay:
            time.sleep(delay)
        if verdict == "pass":
            return {"status": "Success", "run_result": {"status": "Finished", "return_code": 0, "stdout": "ok", "stderr": ""}}, None
        # runtime failure -> result_status -2 (a non-pass)
        return {"status": "Failed", "run_result": {"status": "Finished", "return_code": 1, "stdout": "", "stderr": "boom"}}, None

    return fake_call, calls


def _in_outs(expected):
    return {"inputs": [str(i) for i in range(N)], "outputs": [expected] * N}


def test_short_circuit_cancels_remaining_on_failure(monkeypatch):
    fake_call, calls = _fake_call_factory("fail", delay=0.05)
    monkeypatch.setattr(utils, "call_sandbox_api", fake_call)
    # Semaphore of 1 serialises execution so the first completed case is the
    # first failure -- the rest must bail without calling the sandbox.
    sem = threading.Semaphore(1)
    results, _ = utils.check_correctness(
        "http://sandbox", _in_outs("ok"), "print(1)", timeout=1,
        concurrent_semaphore=sem, retry_on_timeout=False, short_circuit=True,
    )
    assert results.count(True) == 0
    # At most a couple of cases sneak through before stop_event is observed.
    assert len(calls) <= 3, f"expected short-circuit, but {len(calls)}/{N} cases ran"


def test_no_short_circuit_runs_every_case(monkeypatch):
    fake_call, calls = _fake_call_factory("fail")
    monkeypatch.setattr(utils, "call_sandbox_api", fake_call)
    results, _ = utils.check_correctness(
        "http://sandbox", _in_outs("ok"), "print(1)", timeout=1,
        retry_on_timeout=False, short_circuit=False,
    )
    # Default (continuous/pass-rate) behaviour: every case is evaluated.
    assert len(calls) == N


def test_short_circuit_passing_solution_runs_every_case(monkeypatch):
    fake_call, calls = _fake_call_factory("pass")
    monkeypatch.setattr(utils, "call_sandbox_api", fake_call)
    results, _ = utils.check_correctness(
        "http://sandbox", _in_outs("ok"), "print(1)", timeout=1,
        retry_on_timeout=False, short_circuit=True,
    )
    # No failure -> nothing is cancelled -> a correct solution runs all cases.
    assert len(calls) == N
    assert results.count(True) == N


def test_compute_score_binary_verdict(monkeypatch):
    # All pass -> 1.0
    fake_pass, _ = _fake_call_factory("pass")
    monkeypatch.setattr(utils, "call_sandbox_api", fake_pass)
    score, _ = compute_score("http://sandbox", None, 1024, "```python\nprint(1)\n```", _in_outs("ok"), continuous=False)
    assert score == 1.0

    # Any failure -> 0.0 (binary, not pass-rate)
    fake_fail, _ = _fake_call_factory("fail")
    monkeypatch.setattr(utils, "call_sandbox_api", fake_fail)
    score, _ = compute_score("http://sandbox", None, 1024, "```python\nprint(1)\n```", _in_outs("ok"), continuous=False)
    assert score == 0.0
