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
"""Output consistency tests for the sandbox server.

Verifies that identical requests produce byte-identical outputs across
many repeated invocations. This catches non-determinism in the sandbox
infrastructure (temp-dir paths leaking into output, unstable ordering,
race conditions in I/O capture, etc.).

Each test runs the same request N times and asserts that every response
matches the first.
"""

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

def _run_request(request: RunCodeRequest) -> RunCodeResponse:
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    return RunCodeResponse(**response.json())

def _assert_all_identical(results: list[RunCodeResponse], check_stdout=True, check_stderr=False):
    """Assert that all results share the same status, stdout, and optionally stderr."""
    first = results[0]
    for i, r in enumerate(results[1:], start=1):
        assert r.status == first.status, f"Run {i}: status {r.status} != {first.status}"
        if check_stdout:
            assert r.run_result.stdout == first.run_result.stdout, (
                f"Run {i}: stdout differs\n  expected: {first.run_result.stdout!r}\n  got:      {r.run_result.stdout!r}"
            )
        if check_stderr:
            assert r.run_result.stderr == first.run_result.stderr, (
                f"Run {i}: stderr differs\n  expected: {first.run_result.stderr!r}\n  got:      {r.run_result.stderr!r}"
            )

# ---------------------------------------------------------------------------
#  Python: deterministic output
# ---------------------------------------------------------------------------

def test_python_print_consistency():
    """Simple print must produce identical stdout across 30 runs."""
    n = 30
    request = RunCodeRequest(language='python', code='print("deterministic_output_12345")', run_timeout=10)
    results = [_run_request(request) for _ in range(n)]
    _assert_all_identical(results)
    assert results[0].run_result.stdout.strip() == 'deterministic_output_12345'

def test_python_computation_consistency():
    """A pure computation must produce identical results across repeated runs."""
    n = 30
    code = '''
import hashlib
data = "".join(str(i) for i in range(10000))
digest = hashlib.sha256(data.encode()).hexdigest()
print(digest)
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=10)
    results = [_run_request(request) for _ in range(n)]
    _assert_all_identical(results)
    assert len(results[0].run_result.stdout.strip()) == 64

def test_python_multiline_output_consistency():
    """Multi-line structured output must be identical across runs."""
    n = 25
    code = '''
for i in range(20):
    print(f"line_{i:04d}: {i * i}")
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=10)
    results = [_run_request(request) for _ in range(n)]
    _assert_all_identical(results)
    lines = results[0].run_result.stdout.strip().split('\n')
    assert len(lines) == 20

def test_python_sorted_output_consistency():
    """Sorted collection output must be stable across runs."""
    n = 25
    code = '''
import json
data = {chr(65 + i): i * i for i in range(26)}
print(json.dumps(data, sort_keys=True))
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=10)
    results = [_run_request(request) for _ in range(n)]
    _assert_all_identical(results)

def test_python_stdin_consistency():
    """Same stdin must produce the same output every time."""
    n = 25
    code = '''
import sys
lines = sys.stdin.read().strip().split('\\n')
total = sum(int(x) for x in lines)
print(total)
'''
    stdin_data = '\n'.join(str(i) for i in range(1, 51))
    request = RunCodeRequest(language='python', code=code, run_timeout=10, stdin=stdin_data)
    results = [_run_request(request) for _ in range(n)]
    _assert_all_identical(results)
    assert results[0].run_result.stdout.strip() == str(sum(range(1, 51)))

def test_python_numpy_consistency():
    """Seeded NumPy operations must produce identical results."""
    n = 20
    code = '''
import numpy as np
rng = np.random.default_rng(seed=42)
arr = rng.standard_normal(100)
print(f"{arr.mean():.10f}")
print(f"{arr.std():.10f}")
print(f"{arr.sum():.10f}")
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=10)
    results = [_run_request(request) for _ in range(n)]
    _assert_all_identical(results)

# ---------------------------------------------------------------------------
#  C++: deterministic output
# ---------------------------------------------------------------------------

def test_cpp_computation_consistency():
    """A pure C++ computation must produce identical results across runs."""
    n = 20
    code = '''
    #include <iostream>
    int main() {
        long long sum = 0;
        for (int i = 1; i <= 10000; i++) sum += (long long)i * i;
        std::cout << sum << std::endl;
        return 0;
    }
    '''
    request = RunCodeRequest(language='cpp', code=code, run_timeout=10, compile_timeout=15)
    results = [_run_request(request) for _ in range(n)]
    _assert_all_identical(results)
    assert results[0].run_result.stdout.strip() == '333383335000'

def test_cpp_string_output_consistency():
    """Formatted string output in C++ must be stable across runs."""
    n = 20
    code = '''
    #include <iostream>
    #include <iomanip>
    #include <cmath>
    int main() {
        for (int i = 0; i < 10; i++) {
            std::cout << std::fixed << std::setprecision(8)
                      << "val_" << i << " = " << std::sin(i * 0.1) << std::endl;
        }
        return 0;
    }
    '''
    request = RunCodeRequest(language='cpp', code=code, run_timeout=10, compile_timeout=15)
    results = [_run_request(request) for _ in range(n)]
    _assert_all_identical(results)

# ---------------------------------------------------------------------------
#  Bash: deterministic output
# ---------------------------------------------------------------------------

def test_bash_output_consistency():
    """Bash arithmetic and string output must be identical across runs."""
    n = 25
    code = '''
for i in $(seq 1 15); do
    echo "item_${i}: $((i * i))"
done
'''
    request = RunCodeRequest(language='bash', code=code, run_timeout=10)
    results = [_run_request(request) for _ in range(n)]
    _assert_all_identical(results)

# ---------------------------------------------------------------------------
#  Error output consistency
# ---------------------------------------------------------------------------

def _normalize_stderr(stderr: str) -> str:
    """Strip temp directory paths and filenames from stderr so comparisons ignore path variance."""
    import re
    return re.sub(r'/tmp/[^\s"\',:]+\.\w+', '/tmp/TMP_FILE', stderr)

def test_python_error_consistency():
    """The same error must produce the same error type and message across runs."""
    n = 20
    code = 'x = 1 / 0'
    request = RunCodeRequest(language='python', code=code, run_timeout=10)
    results = [_run_request(request) for _ in range(n)]
    first = results[0]
    assert first.status == RunStatus.Failed
    first_normalized = _normalize_stderr(first.run_result.stderr)
    for i, r in enumerate(results[1:], start=1):
        assert r.status == RunStatus.Failed
        assert _normalize_stderr(r.run_result.stderr) == first_normalized, (
            f"Run {i}: stderr differs after path normalization"
        )

def test_cpp_compile_error_consistency():
    """The same compile error must produce the same error message across runs."""
    n = 15
    code = 'int main() { undefined_symbol(); return 0; }'
    request = RunCodeRequest(language='cpp', code=code, run_timeout=10, compile_timeout=15)
    results = [_run_request(request) for _ in range(n)]
    first = results[0]
    assert first.status == RunStatus.Failed
    first_normalized = _normalize_stderr(first.compile_result.stderr)
    for i, r in enumerate(results[1:], start=1):
        assert r.status == RunStatus.Failed
        assert _normalize_stderr(r.compile_result.stderr) == first_normalized, (
            f"Run {i}: compile stderr differs after path normalization"
        )

# ---------------------------------------------------------------------------
#  Return code consistency
# ---------------------------------------------------------------------------

def test_python_exit_code_consistency():
    """sys.exit(N) must return the same code every time."""
    n = 20
    request = RunCodeRequest(language='python', code='import sys; sys.exit(13)', run_timeout=10)
    results = [_run_request(request) for _ in range(n)]
    for r in results:
        assert r.status == RunStatus.Failed
        assert r.run_result.return_code == 13

def test_cpp_return_code_consistency():
    """C++ main returning N must return the same code every time."""
    n = 15
    code = 'int main() { return 7; }'
    request = RunCodeRequest(language='cpp', code=code, run_timeout=10, compile_timeout=15)
    results = [_run_request(request) for _ in range(n)]
    for r in results:
        assert r.status == RunStatus.Failed
        assert r.compile_result.return_code == 0
        assert r.run_result.return_code == 7

# ---------------------------------------------------------------------------
#  File round-trip consistency
# ---------------------------------------------------------------------------

def test_python_file_fetch_consistency():
    """Written files must produce identical base64 content across runs."""
    n = 20
    code = '''
with open("out.txt", "w") as f:
    for i in range(50):
        f.write(f"line {i}: {'x' * (i + 1)}\\n")
print("done")
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=10, fetch_files=['out.txt'])
    results = [_run_request(request) for _ in range(n)]
    _assert_all_identical(results)
    first_file = results[0].files['out.txt']
    for r in results[1:]:
        assert r.files['out.txt'] == first_file

# ---------------------------------------------------------------------------
#  Execution time stability (no pathological variance)
# ---------------------------------------------------------------------------

def test_python_execution_time_stability():
    """Execution times for a trivial program should not vary wildly."""
    n = 20
    request = RunCodeRequest(language='python', code='print("fast")', run_timeout=10)
    results = [_run_request(request) for _ in range(n)]
    times = [r.run_result.execution_time for r in results]
    _assert_all_identical(results)
    avg = sum(times) / len(times)
    # No single run should take more than 10x the average
    for t in times:
        assert t < avg * 10, f"Execution time {t:.3f}s is >10x the average {avg:.3f}s"

# ---------------------------------------------------------------------------
#  Cross-language consistency
# ---------------------------------------------------------------------------

def test_same_algorithm_across_languages():
    """The same algorithm in Python, C++, and Bash must produce the same final answer."""
    n = 10
    expected = '5050'

    py_code = 'print(sum(range(1, 101)))'
    cpp_code = '''
    #include <iostream>
    int main() {
        int s = 0;
        for (int i = 1; i <= 100; i++) s += i;
        std::cout << s << std::endl;
        return 0;
    }
    '''
    bash_code = 'echo $((100 * 101 / 2))'

    for lang, code, extra in [
        ('python', py_code, {}),
        ('cpp', cpp_code, {'compile_timeout': 15}),
        ('bash', bash_code, {}),
    ]:
        request = RunCodeRequest(language=lang, code=code, run_timeout=10, **extra)
        for _ in range(n):
            result = _run_request(request)
            assert result.status == RunStatus.Success
            assert result.run_result.stdout.strip() == expected, (
                f"{lang} run produced {result.run_result.stdout.strip()!r}, expected {expected!r}"
            )
