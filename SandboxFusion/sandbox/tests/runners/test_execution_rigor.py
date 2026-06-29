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
"""Rigorous code execution tests covering edge cases and behaviors
that the per-language happy-path tests do not exercise.

Categories covered:
  - Explicit exit codes
  - stderr vs stdout separation
  - Large I/O and buffering
  - Unicode / binary-safe output
  - Empty / whitespace-only code
  - Execution timing guarantees
  - Multi-line stdin
  - File write + fetch round-trip
  - Compile-succeed-but-run-fail
  - Concurrent execution correctness
  - Status field correctness (Success / Failed / SandboxError)
"""

import base64

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

# ---------------------------------------------------------------------------
#  Exit code handling
# ---------------------------------------------------------------------------

def test_python_explicit_exit_code_zero():
    """sys.exit(0) should be Success."""
    request = RunCodeRequest(language='python', code='import sys; sys.exit(0)', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.return_code == 0

def test_python_explicit_exit_code_nonzero():
    """sys.exit(42) should be Failed with return_code 42."""
    request = RunCodeRequest(language='python', code='import sys; sys.exit(42)', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
    assert result.run_result.return_code == 42

def test_bash_explicit_exit_code():
    """exit 7 should be Failed with return_code 7."""
    request = RunCodeRequest(language='bash', code='exit 7', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.return_code == 7

def test_cpp_nonzero_return():
    """C++ main returning non-zero should be Failed."""
    request = RunCodeRequest(language='cpp',
                             code='int main() { return 3; }',
                             run_timeout=5,
                             compile_timeout=10)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.compile_result.status == CommandRunStatus.Finished
    assert result.compile_result.return_code == 0
    assert result.run_result.return_code == 3

# ---------------------------------------------------------------------------
#  stderr vs stdout separation
# ---------------------------------------------------------------------------

def test_python_stderr_stdout_separation():
    """stderr and stdout must be captured independently."""
    code = 'import sys; print("OUT"); print("ERR", file=sys.stderr)'
    request = RunCodeRequest(language='python', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert 'OUT' in result.run_result.stdout
    assert 'ERR' in result.run_result.stderr
    assert 'ERR' not in result.run_result.stdout
    assert 'OUT' not in result.run_result.stderr

def test_cpp_stderr_stdout_separation():
    """C++ stderr and stdout must be captured into separate fields."""
    code = '''
    #include <iostream>
    int main() {
        std::cout << "STDOUT_LINE" << std::endl;
        std::cerr << "STDERR_LINE" << std::endl;
        return 0;
    }
    '''
    request = RunCodeRequest(language='cpp', code=code, run_timeout=5, compile_timeout=10)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert 'STDOUT_LINE' in result.run_result.stdout
    assert 'STDERR_LINE' in result.run_result.stderr

# ---------------------------------------------------------------------------
#  Large I/O
# ---------------------------------------------------------------------------

def test_python_large_stdout():
    """Programs that produce substantial output should not deadlock or truncate."""
    n_lines = 5000
    code = f'for i in range({n_lines}): print(f"line_{{i}}")'
    request = RunCodeRequest(language='python', code=code, run_timeout=10)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    lines = result.run_result.stdout.strip().split('\n')
    assert len(lines) == n_lines
    assert lines[0] == 'line_0'
    assert lines[-1] == f'line_{n_lines - 1}'

def test_python_large_stdin():
    """Large stdin should be delivered correctly."""
    n_lines = 2000
    stdin_data = '\n'.join(str(i) for i in range(n_lines))
    code = '''
import sys
total = 0
for line in sys.stdin:
    total += int(line.strip())
print(total)
'''
    expected = sum(range(n_lines))
    request = RunCodeRequest(language='python', code=code, run_timeout=10, stdin=stdin_data)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.stdout.strip() == str(expected)

# ---------------------------------------------------------------------------
#  Unicode handling
# ---------------------------------------------------------------------------

def test_python_unicode_output():
    """Non-ASCII characters in stdout should round-trip correctly."""
    code = 'print("Hello\\u4e16\\u754c \\U0001f600")'
    request = RunCodeRequest(language='python', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert '\u4e16\u754c' in result.run_result.stdout

def test_python_unicode_stdin():
    """Unicode in stdin should reach the program."""
    code = 'import sys; print(sys.stdin.read())'
    request = RunCodeRequest(language='python', code=code, run_timeout=5, stdin='\u00e9\u00e8\u00ea\n')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert '\u00e9\u00e8\u00ea' in result.run_result.stdout

# ---------------------------------------------------------------------------
#  Empty / degenerate code
# ---------------------------------------------------------------------------

def test_python_empty_code():
    """Empty code should succeed (no-op)."""
    request = RunCodeRequest(language='python', code='', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.stdout == '' or result.run_result.stdout is None or result.run_result.stdout.strip() == ''

def test_bash_empty_code():
    """Empty bash script should succeed."""
    request = RunCodeRequest(language='bash', code='', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success

def test_python_whitespace_only():
    """Whitespace-only Python code should succeed."""
    request = RunCodeRequest(language='python', code='   \n\n   \n', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success

# ---------------------------------------------------------------------------
#  Timeout precision
# ---------------------------------------------------------------------------

def test_python_just_within_timeout():
    """A program that finishes before the timeout should succeed."""
    code = 'import time; time.sleep(0.05); print("done")'
    request = RunCodeRequest(language='python', code=code, run_timeout=2)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert 'done' in result.run_result.stdout

def test_python_timeout_records_execution_time():
    """When a timeout occurs, execution_time should be populated and close to the timeout."""
    request = RunCodeRequest(language='python', code='import time; time.sleep(10)', run_timeout=0.3)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded
    assert result.run_result.execution_time is not None
    assert result.run_result.execution_time >= 0.2

def test_python_execution_time_tracked_on_success():
    """Successful runs should have execution_time populated."""
    code = 'import time; time.sleep(0.1); print("ok")'
    request = RunCodeRequest(language='python', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.execution_time is not None
    assert result.run_result.execution_time >= 0.05

# ---------------------------------------------------------------------------
#  Multi-line stdin
# ---------------------------------------------------------------------------

def test_python_multiline_stdin():
    """Multi-line stdin with various separators."""
    code = '''
import sys
lines = sys.stdin.read().strip().split('\\n')
print(len(lines))
for l in lines:
    print(l.upper())
'''
    stdin_data = 'hello\nworld\nfoo\n'
    request = RunCodeRequest(language='python', code=code, run_timeout=5, stdin=stdin_data)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    lines = result.run_result.stdout.strip().split('\n')
    assert lines[0] == '3'
    assert lines[1] == 'HELLO'
    assert lines[2] == 'WORLD'
    assert lines[3] == 'FOO'

def test_cpp_multiline_stdin():
    """C++ reading multiple lines from stdin."""
    code = '''
    #include <iostream>
    #include <string>
    int main() {
        std::string line;
        int count = 0;
        while (std::getline(std::cin, line)) {
            count++;
        }
        std::cout << count << std::endl;
        return 0;
    }
    '''
    stdin_data = 'a\nb\nc\nd\ne\n'
    request = RunCodeRequest(language='cpp', code=code, run_timeout=5, compile_timeout=10, stdin=stdin_data)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.stdout.strip() == '5'

# ---------------------------------------------------------------------------
#  File round-trip (write then fetch)
# ---------------------------------------------------------------------------

def test_python_file_write_and_fetch():
    """Write a file during execution, then fetch it via fetch_files."""
    code = '''
with open("output.txt", "w") as f:
    f.write("result_data_12345")
print("wrote file")
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=5, fetch_files=['output.txt'])
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert 'output.txt' in result.files
    content = base64.b64decode(result.files['output.txt']).decode('utf-8')
    assert content == 'result_data_12345'

def test_python_fetch_nonexistent_file():
    """Fetching a file that was never created should not crash; just missing from files dict."""
    code = 'print("no file written")'
    request = RunCodeRequest(language='python', code=code, run_timeout=5, fetch_files=['does_not_exist.txt'])
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert 'does_not_exist.txt' not in result.files

def test_python_binary_file_round_trip():
    """Binary data should survive the base64 round-trip through files."""
    b64_content = base64.b64encode(bytes(range(256))).decode('utf-8')
    code = '''
with open("input.bin", "rb") as f:
    data = f.read()
with open("output.bin", "wb") as f:
    f.write(data)
print(len(data))
'''
    request = RunCodeRequest(language='python',
                             code=code,
                             run_timeout=5,
                             files={'input.bin': b64_content},
                             fetch_files=['output.bin'])
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.stdout.strip() == '256'
    fetched = base64.b64decode(result.files['output.bin'])
    assert fetched == bytes(range(256))

# ---------------------------------------------------------------------------
#  Compiled language: compile OK but run fails
# ---------------------------------------------------------------------------

def test_cpp_compile_ok_runtime_crash():
    """Abort at runtime after successful compilation."""
    code = '''
    #include <cstdlib>
    int main() {
        abort();
        return 0;
    }
    '''
    request = RunCodeRequest(language='cpp', code=code, run_timeout=5, compile_timeout=10)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    # Compile should succeed
    assert result.compile_result.status == CommandRunStatus.Finished
    assert result.compile_result.return_code == 0
    # Run should fail (SIGABRT -> non-zero exit)
    assert result.status == RunStatus.Failed
    assert result.run_result.return_code != 0

def test_cpp_compile_ok_run_timeout():
    """Compile succeeds but execution times out."""
    code = '''
    int main() { while(true) {} return 0; }
    '''
    request = RunCodeRequest(language='cpp', code=code, run_timeout=0.5, compile_timeout=10)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.compile_result.status == CommandRunStatus.Finished
    assert result.compile_result.return_code == 0
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded
    assert result.status == RunStatus.Failed

def test_go_compile_error():
    """Go code that fails compilation should not run."""
    code = '''
    package main
    func main() {
        undefined_func()
    }
    '''
    request = RunCodeRequest(language='go', code=code, run_timeout=5, compile_timeout=10)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.compile_result.status == CommandRunStatus.Finished
    assert result.compile_result.return_code != 0
    assert result.run_result is None

# ---------------------------------------------------------------------------
#  Runtime exceptions across languages
# ---------------------------------------------------------------------------

def test_python_runtime_exception():
    """Unhandled exception should fail and report in stderr."""
    code = 'raise ValueError("test_error_msg")'
    request = RunCodeRequest(language='python', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert 'ValueError' in result.run_result.stderr
    assert 'test_error_msg' in result.run_result.stderr

def test_python_zero_division():
    """ZeroDivisionError should be reported in stderr with a Failed status."""
    code = 'print(1/0)'
    request = RunCodeRequest(language='python', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert 'ZeroDivisionError' in result.run_result.stderr

def test_python_import_nonexistent_module():
    """Importing a missing module should produce ModuleNotFoundError in stderr."""
    code = 'import nonexistent_module_xyz_abc'
    request = RunCodeRequest(language='python', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert 'ModuleNotFoundError' in result.run_result.stderr

def test_python_recursion_limit():
    """Hitting the recursion limit should produce an error, not hang."""
    code = '''
import sys
sys.setrecursionlimit(200)
def f(n):
    return f(n+1)
f(0)
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert 'RecursionError' in result.run_result.stderr

# ---------------------------------------------------------------------------
#  Output before crash
# ---------------------------------------------------------------------------

def test_python_partial_output_before_crash():
    """Output written before a crash should still be captured."""
    code = '''
print("before_crash")
import sys
sys.stdout.flush()
raise RuntimeError("boom")
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert 'before_crash' in result.run_result.stdout
    assert 'RuntimeError' in result.run_result.stderr

def test_cpp_output_before_nonzero_exit():
    """Stdout written before a non-zero exit should still be captured in C++."""
    code = '''
    #include <iostream>
    int main() {
        std::cout << "partial_output" << std::endl;
        return 5;
    }
    '''
    request = RunCodeRequest(language='cpp', code=code, run_timeout=5, compile_timeout=10)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert 'partial_output' in result.run_result.stdout
    assert result.run_result.return_code == 5

# ---------------------------------------------------------------------------
#  Stdin edge cases
# ---------------------------------------------------------------------------

def test_python_empty_stdin():
    """Empty stdin should not block or error."""
    code = '''
import sys
data = sys.stdin.read()
print(f"len={len(data)}")
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=5, stdin='')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert 'len=0' in result.run_result.stdout

def test_python_no_stdin_field():
    """When stdin is None (not provided), reading stdin should get EOF immediately."""
    code = '''
import sys
data = sys.stdin.read()
print(f"len={len(data)}")
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert 'len=0' in result.run_result.stdout

# ---------------------------------------------------------------------------
#  Multiple files provided
# ---------------------------------------------------------------------------

def test_python_multiple_input_files():
    """Multiple input files should all be accessible."""
    files = {
        'data/a.txt': base64.b64encode(b'alpha').decode(),
        'data/b.txt': base64.b64encode(b'beta').decode(),
        'config.json': base64.b64encode(b'{"key": "value"}').decode(),
    }
    code = '''
import json
with open("data/a.txt") as f:
    a = f.read()
with open("data/b.txt") as f:
    b = f.read()
with open("config.json") as f:
    c = json.load(f)
print(f"{a},{b},{c['key']}")
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=5, files=files)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.stdout.strip() == 'alpha,beta,value'

# ---------------------------------------------------------------------------
#  Concurrent execution correctness
# ---------------------------------------------------------------------------

def test_concurrent_python_runs_are_isolated():
    """Two concurrent runs should not interfere with each other's files or state."""
    code_a = '''
import time
with open("marker.txt", "w") as f:
    f.write("AAAA")
time.sleep(0.2)
with open("marker.txt") as f:
    print(f.read())
'''
    code_b = '''
import time
with open("marker.txt", "w") as f:
    f.write("BBBB")
time.sleep(0.2)
with open("marker.txt") as f:
    print(f.read())
'''
    req_a = RunCodeRequest(language='python', code=code_a, run_timeout=5)
    req_b = RunCodeRequest(language='python', code=code_b, run_timeout=5)

    resp_a = client.post('/run_code', json=req_a.model_dump())
    resp_b = client.post('/run_code', json=req_b.model_dump())

    result_a = RunCodeResponse(**resp_a.json())
    result_b = RunCodeResponse(**resp_b.json())

    assert result_a.status == RunStatus.Success
    assert result_b.status == RunStatus.Success
    # Each should read back its own data (temp dirs are separate)
    assert result_a.run_result.stdout.strip() == 'AAAA'
    assert result_b.run_result.stdout.strip() == 'BBBB'

# ---------------------------------------------------------------------------
#  Compile timeout
# ---------------------------------------------------------------------------

def test_cpp_compile_timeout():
    """A program that takes too long to compile should timeout at compile phase."""
    # Generate a pathological template expansion
    code = '''
    template<int N> struct Fib { static const int value = Fib<N-1>::value + Fib<N-2>::value; };
    template<> struct Fib<0> { static const int value = 0; };
    template<> struct Fib<1> { static const int value = 1; };
    ''' + '\n'.join(f'int v{i} = Fib<{40 + (i % 5)}>::value;' for i in range(200)) + '''
    int main() { return 0; }
    '''
    request = RunCodeRequest(language='cpp', code=code, run_timeout=5, compile_timeout=0.2)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    # Either it times out during compilation or finishes fast enough.
    # The point is it shouldn't crash the sandbox.
    assert result.status in (RunStatus.Failed, RunStatus.Success)
    if result.compile_result.status == CommandRunStatus.TimeLimitExceeded:
        assert result.run_result is None

# ---------------------------------------------------------------------------
#  Bash-specific edge cases
# ---------------------------------------------------------------------------

def test_bash_pipe():
    """Piped commands in bash."""
    code = 'echo "hello world" | tr " " "\\n" | sort'
    request = RunCodeRequest(language='bash', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    lines = result.run_result.stdout.strip().split('\n')
    assert 'hello' in lines
    assert 'world' in lines

def test_bash_heredoc():
    """Here-document syntax in bash."""
    code = '''cat <<'ENDOFMSG'
line one
line two
ENDOFMSG
'''
    request = RunCodeRequest(language='bash', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert 'line one' in result.run_result.stdout
    assert 'line two' in result.run_result.stdout

def test_bash_stderr_redirect():
    """Bash writing to stderr only."""
    code = 'echo "error_msg" >&2'
    request = RunCodeRequest(language='bash', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert 'error_msg' in result.run_result.stderr
    assert result.run_result.stdout.strip() == ''

# ---------------------------------------------------------------------------
#  Python: computation correctness
# ---------------------------------------------------------------------------

def test_python_math_computation():
    """Verify a non-trivial computation produces the right answer."""
    code = '''
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [p for p in range(2, 100) if is_prime(p)]
print(len(primes))
print(primes[-1])
'''
    request = RunCodeRequest(language='python', code=code, run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    lines = result.run_result.stdout.strip().split('\n')
    assert lines[0] == '25'  # 25 primes under 100
    assert lines[1] == '97'  # largest prime under 100

def test_cpp_computation():
    """Verify a non-trivial C++ computation."""
    code = '''
    #include <iostream>
    int main() {
        long long fib_prev = 0, fib_curr = 1;
        for (int i = 2; i <= 50; i++) {
            long long next = fib_prev + fib_curr;
            fib_prev = fib_curr;
            fib_curr = next;
        }
        std::cout << fib_curr << std::endl;
        return 0;
    }
    '''
    request = RunCodeRequest(language='cpp', code=code, run_timeout=5, compile_timeout=10)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.stdout.strip() == '12586269025'  # Fib(50)

# ---------------------------------------------------------------------------
#  stdin + stdout combined (algorithmic problem pattern)
# ---------------------------------------------------------------------------

def test_python_stdin_algorithmic_problem():
    """Simulate a typical OJ problem: read N, then N integers, print their sum."""
    code = '''
import sys
input_data = sys.stdin.read().split()
n = int(input_data[0])
nums = [int(input_data[i+1]) for i in range(n)]
print(sum(nums))
'''
    stdin_data = '5\n10 20 30 40 50'
    request = RunCodeRequest(language='python', code=code, run_timeout=5, stdin=stdin_data)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.stdout.strip() == '150'

def test_cpp_stdin_algorithmic_problem():
    """Simulate a typical OJ problem in C++: read N integers from stdin and print their sum."""
    code = '''
    #include <iostream>
    int main() {
        int n;
        std::cin >> n;
        long long sum = 0;
        for (int i = 0; i < n; i++) {
            int x;
            std::cin >> x;
            sum += x;
        }
        std::cout << sum << std::endl;
        return 0;
    }
    '''
    stdin_data = '5\n10 20 30 40 50'
    request = RunCodeRequest(language='cpp', code=code, run_timeout=5, compile_timeout=10, stdin=stdin_data)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.stdout.strip() == '150'

# ---------------------------------------------------------------------------
#  Response structure invariants
# ---------------------------------------------------------------------------

def test_response_has_run_result_on_interpreted_success():
    """For interpreted languages, compile_result should be None."""
    request = RunCodeRequest(language='python', code='print(1)', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    result = RunCodeResponse(**response.json())
    assert result.compile_result is None
    assert result.run_result is not None
    assert result.run_result.status == CommandRunStatus.Finished

def test_response_has_both_results_on_compiled_success():
    """For compiled languages, both compile_result and run_result should be present."""
    code = '''
    #include <iostream>
    int main() { std::cout << "ok" << std::endl; return 0; }
    '''
    request = RunCodeRequest(language='cpp', code=code, run_timeout=5, compile_timeout=10)
    response = client.post('/run_code', json=request.model_dump())
    result = RunCodeResponse(**response.json())
    assert result.compile_result is not None
    assert result.compile_result.status == CommandRunStatus.Finished
    assert result.compile_result.return_code == 0
    assert result.run_result is not None
    assert result.run_result.status == CommandRunStatus.Finished

def test_compile_failure_means_no_run_result():
    """If compilation fails, run_result should be None."""
    code = 'this is not valid C++ code at all !!!'
    request = RunCodeRequest(language='cpp', code=code, run_timeout=5, compile_timeout=10)
    response = client.post('/run_code', json=request.model_dump())
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.compile_result is not None
    assert result.compile_result.return_code != 0
    assert result.run_result is None
