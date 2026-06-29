# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""RL-reward-loop load tests for the sandbox server.

Replays the request pattern an RL trainer (verl) generates against the
sandbox, so capacity regressions are caught here instead of half an hour
into a GPU job.  The historical failure mode these tests guard against:
training rewards collapse to zero at around step 4-5 because the server
slowly loses execution slots (e.g. the stdin-drain leak: a solution that
stays alive without consuming a large stdin used to block the handler
before the wall-clock kill was armed) until queue waits exceed the
client's 30s read timeout and every request fails while a single trivial
probe still answers in milliseconds.

The client behavior mirrors verl's reward loop: ~48 concurrent in-flight
requests (8 workers x max_concurrent=6), a 30s read timeout
(compile_timeout + run_timeout + 10s buffer), and a workload mix that
includes correct, wrong, crashing, time-limit-exceeded, and
large-stdin-not-consumed ("poison") solutions.

These tests are marked ``stress`` and excluded from the default suite;
run them with ``make test-rl-load``.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pytest

from sandbox.tests.client import client

pytestmark = pytest.mark.stress

# What verl uses: compile_timeout(10) + run_timeout(10) + API buffer(10).
CLIENT_TIMEOUT = 30
# Keep individual executions short so the whole storm stays in the minutes
# range; the failure modes under test do not depend on the timeout value.
RUN_TIMEOUT = 2
# max_concurrency in configs/docker_bindroot.yaml.
SERVER_SLOTS = 32
# verl: 8 RewardLoopWorkers x max_concurrent=6.
CLIENT_STREAMS = 48

# Larger than the 64 KiB pipe buffer, like competitive-programming inputs.
BIG_STDIN = 'x' * (1024 * 1024)

# (kind, payload, allowed run_result statuses) -- the mix an RL batch
# produces.  "Allowed" statuses describe a healthy server; a client-side
# timeout is never acceptable for any of them.
WORKLOAD = [
    ('ok', {
        'language': 'python',
        'code': 'a, b = map(int, input().split())\nprint(a + b)',
        'stdin': '40 2',
        'run_timeout': RUN_TIMEOUT,
    }, {'Finished'}),
    ('wrong_answer', {
        'language': 'python',
        'code': 'print(43)',
        'stdin': '40 2',
        'run_timeout': RUN_TIMEOUT,
    }, {'Finished'}),
    ('runtime_error', {
        'language': 'python',
        'code': 'print(1 // 0)',
        'stdin': '',
        'run_timeout': RUN_TIMEOUT,
    }, {'Finished'}),
    ('big_stdin_consumed', {
        'language': 'python',
        'code': 'import sys\nprint(len(sys.stdin.read()))',
        'stdin': BIG_STDIN,
        'run_timeout': RUN_TIMEOUT + 4,
    }, {'Finished'}),
    ('tle_busy', {
        'language': 'python',
        'code': 'while True: pass',
        'stdin': '',
        'run_timeout': RUN_TIMEOUT,
    }, {'TimeLimitExceeded'}),
    # The slot-leak regression case: stays alive, never reads its 1 MB stdin.
    ('tle_poison_stdin', {
        'language': 'python',
        'code': 'import time\ntime.sleep(60)',
        'stdin': BIG_STDIN,
        'run_timeout': RUN_TIMEOUT,
    }, {'TimeLimitExceeded'}),
]


def _post(payload, timeout=CLIENT_TIMEOUT):
    """POST one /run_code request, mirroring verl's client semantics.

    Returns a dict with ``elapsed`` plus either ``timeout=True`` (the
    client gave up, like verl's requests.post read timeout) or the parsed
    ``status``/``run_status`` of the response.
    """
    start = time.monotonic()
    try:
        resp = client.post('/run_code', json=payload, timeout=timeout)
    except httpx.TimeoutException:
        return {'timeout': True, 'elapsed': time.monotonic() - start}
    elapsed = time.monotonic() - start
    assert resp.status_code == 200, f'HTTP {resp.status_code}: {resp.text[:200]}'
    body = resp.json()
    run_result = body.get('run_result') or {}
    return {
        'timeout': False,
        'elapsed': elapsed,
        'status': body.get('status'),
        'run_status': run_result.get('status'),
        'stdout': run_result.get('stdout'),
    }


def _liveness_probe(budget=15):
    """One trivial execution; a healthy server answers in well under 1s.

    The generous budget only absorbs shared-login-node jitter -- anything
    slower means the execution queue is already pathological.
    """
    result = _post({'language': 'python', 'code': 'print(40 + 2)', 'run_timeout': 5}, timeout=budget)
    return result


def _capacity_check():
    """All SERVER_SLOTS trivial requests at once must clear quickly.

    With leaked slots the burst still completes (trivial requests drain
    fast through whatever is left), so this is a coarse check; the strong
    assertions are the per-request ones in the tests themselves.
    """
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=SERVER_SLOTS) as pool:
        futures = [
            pool.submit(_post, {'language': 'python', 'code': 'print(40 + 2)', 'run_timeout': 5})
            for _ in range(SERVER_SLOTS)
        ]
        results = [f.result() for f in futures]
    wall = time.monotonic() - start
    timeouts = [r for r in results if r['timeout']]
    assert not timeouts, f'{len(timeouts)}/{SERVER_SLOTS} trivial requests timed out after the load'
    assert wall < 20, f'{SERVER_SLOTS} trivial requests took {wall:.1f}s; expected < 20s'


def test_rl_load_storm():
    """Mixed RL batch at full client concurrency: nothing may time out.

    48 streams x 6 requests cycle through the workload mix while a probe
    thread checks server liveness every 8 seconds.  Every request must
    come back with its expected run status within verl's 30s budget --
    one client-side timeout here is one poisoned reward in training.
    """
    reqs_per_stream = 6
    requests_total = []
    for stream in range(CLIENT_STREAMS):
        for i in range(reqs_per_stream):
            requests_total.append(WORKLOAD[(stream + i) % len(WORKLOAD)])

    probe_results = []
    storm_done = []

    def prober():
        while not storm_done:
            probe_results.append(_liveness_probe())
            for _ in range(16):
                if storm_done:
                    return
                time.sleep(0.5)

    with ThreadPoolExecutor(max_workers=CLIENT_STREAMS + 1) as pool:
        probe_future = pool.submit(prober)
        futures = {pool.submit(_post, payload): (kind, expected) for kind, payload, expected in requests_total}
        failures = []
        for future in as_completed(futures):
            kind, expected = futures[future]
            r = future.result()
            if r['timeout']:
                failures.append(f'{kind}: client timeout after {r["elapsed"]:.1f}s')
            elif r['run_status'] not in expected:
                failures.append(f'{kind}: run_status={r["run_status"]} (expected {expected})')
        storm_done.append(True)
        probe_future.result()

    assert not failures, 'storm failures:\n' + '\n'.join(failures[:20])
    probe_timeouts = [p for p in probe_results if p['timeout']]
    assert not probe_timeouts, (f'{len(probe_timeouts)}/{len(probe_results)} liveness probes timed out '
                                'while the storm was running')
    _capacity_check()


def test_rl_load_poison_stdin_no_slot_leak():
    """Regression for the stdin-drain slot leak.

    A wave of solutions that stay alive without consuming a 1 MB stdin
    must all come back TimeLimitExceeded within the client budget.  On
    the buggy server each of these hung its handler forever (the drain
    blocked before the wall-clock kill was armed), permanently consuming
    an execution slot per request.
    """
    n_poison = 12
    _, payload, _ = next(w for w in WORKLOAD if w[0] == 'tle_poison_stdin')

    with ThreadPoolExecutor(max_workers=n_poison) as pool:
        results = [f.result() for f in [pool.submit(_post, payload) for _ in range(n_poison)]]

    hung = [r for r in results if r['timeout']]
    assert not hung, (f'{len(hung)}/{n_poison} poison requests hung past the {CLIENT_TIMEOUT}s client '
                      'timeout: the server is not enforcing the run timeout during stdin delivery')
    bad = [r['run_status'] for r in results if r['run_status'] != 'TimeLimitExceeded']
    assert not bad, f'poison requests should be TimeLimitExceeded, got: {bad}'
    _capacity_check()


def test_rl_load_kill_churn():
    """Sustained kill-path pressure: waves of busy-loop TLE executions.

    Three back-to-back waves of SERVER_SLOTS concurrent infinite loops
    exercise process-tree kill and reaping under full saturation (the
    pattern that corrupted the fakeroot daemon historically).  Every
    execution must TLE cleanly and the server must stay responsive
    between waves.
    """
    _, payload, _ = next(w for w in WORKLOAD if w[0] == 'tle_busy')

    for wave in range(3):
        with ThreadPoolExecutor(max_workers=SERVER_SLOTS) as pool:
            results = [f.result() for f in [pool.submit(_post, payload) for _ in range(SERVER_SLOTS)]]
        hung = [r for r in results if r['timeout']]
        assert not hung, f'wave {wave}: {len(hung)}/{SERVER_SLOTS} TLE requests hung past the client timeout'
        bad = [r['run_status'] for r in results if r['run_status'] != 'TimeLimitExceeded']
        assert not bad, f'wave {wave}: expected TimeLimitExceeded, got: {bad}'
        probe = _liveness_probe()
        assert not probe['timeout'], f'liveness probe timed out after wave {wave}'

    _capacity_check()
