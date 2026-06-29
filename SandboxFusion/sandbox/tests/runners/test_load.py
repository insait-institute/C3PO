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
"""Load and concurrency tests for the sandbox server.

Verifies that the server handles parallel requests correctly under
varying levels of concurrency, that no requests are silently dropped
or corrupted, and that mixed-language workloads complete without
interference.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

def _run_request(request: RunCodeRequest) -> RunCodeResponse:
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    return RunCodeResponse(**response.json())

# ---------------------------------------------------------------------------
#  Parallel request handling
# ---------------------------------------------------------------------------

def test_parallel_python_requests():
    """Fire 20 identical Python requests in parallel; all must succeed with correct output."""
    n = 20
    request = RunCodeRequest(language='python', code='print(42 * 42)', run_timeout=10)
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_run_request, request) for _ in range(n)]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == n
    for result in results:
        assert result.status == RunStatus.Success
        assert result.run_result.stdout.strip() == '1764'

def test_parallel_cpp_requests():
    """Fire 10 identical C++ requests in parallel; all must compile and produce correct output."""
    n = 10
    code = '''
    #include <iostream>
    int main() {
        int sum = 0;
        for (int i = 1; i <= 100; i++) sum += i;
        std::cout << sum << std::endl;
        return 0;
    }
    '''
    request = RunCodeRequest(language='cpp', code=code, run_timeout=10, compile_timeout=15)
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_run_request, request) for _ in range(n)]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == n
    for result in results:
        assert result.status == RunStatus.Success
        assert result.compile_result.status == CommandRunStatus.Finished
        assert result.run_result.stdout.strip() == '5050'

def test_parallel_requests_with_unique_inputs():
    """Each parallel request receives a distinct input and must produce the matching output."""
    n = 20
    requests = [
        RunCodeRequest(language='python', code=f'print({i} ** 2)', run_timeout=10) for i in range(n)
    ]
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(_run_request, req): i for i, req in enumerate(requests)}
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            assert result.status == RunStatus.Success
            assert result.run_result.stdout.strip() == str(i ** 2)

def test_parallel_requests_with_stdin():
    """Parallel requests each with different stdin must route input correctly."""
    n = 15
    code = 'import sys; n = int(sys.stdin.read().strip()); print(n * 3)'
    requests = [RunCodeRequest(language='python', code=code, run_timeout=10, stdin=str(i)) for i in range(n)]
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(_run_request, req): i for i, req in enumerate(requests)}
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            assert result.status == RunStatus.Success
            assert result.run_result.stdout.strip() == str(i * 3)

# ---------------------------------------------------------------------------
#  Mixed-language parallel workload
# ---------------------------------------------------------------------------

def test_mixed_language_parallel():
    """Run Python, C++, and Bash requests concurrently; all must succeed."""
    requests_and_expected = [
        (RunCodeRequest(language='python', code='print("py_ok")', run_timeout=10), 'py_ok'),
        (RunCodeRequest(
            language='cpp',
            code='#include <iostream>\nint main() { std::cout << "cpp_ok" << std::endl; return 0; }',
            run_timeout=10,
            compile_timeout=15,
        ), 'cpp_ok'),
        (RunCodeRequest(language='bash', code='echo "bash_ok"', run_timeout=10), 'bash_ok'),
        (RunCodeRequest(language='python', code='print(2 + 2)', run_timeout=10), '4'),
        (RunCodeRequest(language='bash', code='echo $((7 * 6))', run_timeout=10), '42'),
    ]
    # Repeat the mix to increase concurrency
    workload = requests_and_expected * 3

    with ThreadPoolExecutor(max_workers=len(workload)) as pool:
        futures = {pool.submit(_run_request, req): expected for req, expected in workload}
        for future in as_completed(futures):
            expected = futures[future]
            result = future.result()
            assert result.status == RunStatus.Success
            assert result.run_result.stdout.strip() == expected

# ---------------------------------------------------------------------------
#  Sustained sequential load
# ---------------------------------------------------------------------------

def test_sustained_sequential_load():
    """50 sequential requests should all succeed without degradation or errors."""
    n = 50
    for i in range(n):
        request = RunCodeRequest(language='python', code=f'print({i})', run_timeout=10)
        result = _run_request(request)
        assert result.status == RunStatus.Success
        assert result.run_result.stdout.strip() == str(i)

# ---------------------------------------------------------------------------
#  Parallel failures should not poison other requests
# ---------------------------------------------------------------------------

def test_parallel_mix_of_success_and_failure():
    """Requests that fail (syntax errors, timeouts) must not affect concurrent successful ones."""
    good_code = 'print("success")'
    bad_syntax = 'def :'
    timeout_code = 'import time; time.sleep(30)'

    requests = []
    for _ in range(6):
        requests.append(('success', RunCodeRequest(language='python', code=good_code, run_timeout=10)))
    for _ in range(3):
        requests.append(('syntax_error', RunCodeRequest(language='python', code=bad_syntax, run_timeout=10)))
    for _ in range(3):
        requests.append(('timeout', RunCodeRequest(language='python', code=timeout_code, run_timeout=0.3)))

    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        futures = {pool.submit(_run_request, req): kind for kind, req in requests}
        for future in as_completed(futures):
            kind = futures[future]
            result = future.result()
            if kind == 'success':
                assert result.status == RunStatus.Success
                assert result.run_result.stdout.strip() == 'success'
            elif kind == 'syntax_error':
                assert result.status == RunStatus.Failed
                assert 'SyntaxError' in result.run_result.stderr
            elif kind == 'timeout':
                assert result.status == RunStatus.Failed
                assert result.run_result.status == CommandRunStatus.TimeLimitExceeded

# ---------------------------------------------------------------------------
#  No request drops under burst
# ---------------------------------------------------------------------------

def test_burst_all_responses_received():
    """Fire a burst of 30 requests and verify we get exactly 30 valid responses."""
    n = 30
    request = RunCodeRequest(language='python', code='print("ping")', run_timeout=10)
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(_run_request, request) for _ in range(n)]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == n
    success_count = sum(1 for r in results if r.status == RunStatus.Success)
    assert success_count == n

# ---------------------------------------------------------------------------
#  File isolation under parallel load
# ---------------------------------------------------------------------------

def test_parallel_file_isolation():
    """Parallel requests writing/reading files must not see each other's data."""
    n = 15
    codes = []
    for i in range(n):
        codes.append(f'''
with open("data.txt", "w") as f:
    f.write("{i}")
import time; time.sleep(0.05)
with open("data.txt") as f:
    print(f.read())
''')

    requests = [RunCodeRequest(language='python', code=code, run_timeout=10) for code in codes]
    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = {pool.submit(_run_request, req): i for i, req in enumerate(requests)}
        for future in as_completed(futures):
            i = futures[future]
            result = future.result()
            assert result.status == RunStatus.Success
            assert result.run_result.stdout.strip() == str(i)
