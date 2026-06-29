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
"""Rigorous end-to-end submission tests exercising the full
/submit pipeline: code extraction -> code execution -> output comparison.

Uses inline test_cases so tests are self-contained (no database dependency).

Categories covered:
  - Correct submission (accepted)
  - Wrong-answer submission (rejected)
  - Runtime error during evaluation
  - Compilation error during evaluation
  - Timeout during evaluation
  - Code extraction from markdown fences
  - Code extraction from raw code (no fences)
  - Empty / garbage completion
  - Multiple test cases (all must pass)
  - Partial failure across test cases
  - Float tolerance in output comparison
  - Case-insensitive comparison
  - Trailing whitespace / newline tolerance
  - run_all_cases extra flag
  - Multi-language submissions
  - EvalResult structure
"""

from sandbox.datasets.types import EvalResult, GeneralStdioTest, TestConfig, SubmitRequest

from sandbox.tests.client import client

def _make_cases(pairs):
    """Helper to build GeneralStdioTest list from (stdin, stdout) tuples."""
    return [GeneralStdioTest(input={'stdin': p[0]}, output={'stdout': p[1]}) for p in pairs]

# ---------------------------------------------------------------------------
#  Correct submissions (accepted)
# ---------------------------------------------------------------------------

async def test_submit_python_correct_single_case():
    """A Python submission that produces the exact expected output should be accepted."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('3\n', '9\n')]),
        completion='```python\nimport sys\nn = int(sys.stdin.read())\nprint(n * n)\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True
    assert len(result.tests) == 1
    assert result.tests[0].passed is True

async def test_submit_python_correct_multiple_cases():
    """All test cases must pass for accepted=True."""
    cases = _make_cases([('2\n', '4\n'), ('5\n', '25\n'), ('0\n', '0\n'), ('-3\n', '9\n')])
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=cases,
        completion='```python\nimport sys\nn = int(sys.stdin.read())\nprint(n * n)\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True
    assert len(result.tests) == 4
    assert all(t.passed for t in result.tests)

async def test_submit_cpp_correct():
    """A C++ submission through the full pipeline."""
    cpp_code = '''```cpp
#include <iostream>
#include <string>
#include <algorithm>
int main() {
    std::string s;
    std::getline(std::cin, s);
    for (auto &c : s) c = toupper(c);
    std::cout << s << std::endl;
    return 0;
}
```'''
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='cpp'),
        test_cases=_make_cases([('hello\n', 'HELLO\n')]),
        completion=cpp_code,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True

# ---------------------------------------------------------------------------
#  Wrong-answer submissions
# ---------------------------------------------------------------------------

async def test_submit_python_wrong_answer():
    """Incorrect output should be rejected."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('3\n', '9\n')]),
        completion='```python\nimport sys\nn = int(sys.stdin.read())\nprint(n + n)\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False

async def test_submit_partial_failure():
    """If one of multiple test cases fails, accepted should be False."""
    cases = _make_cases([('2\n', '4\n'), ('3\n', '9\n'), ('4\n', '99\n')])  # last is wrong expected
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=cases,
        completion='```python\nimport sys\nn = int(sys.stdin.read())\nprint(n * n)\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False

# ---------------------------------------------------------------------------
#  Runtime errors
# ---------------------------------------------------------------------------

async def test_submit_python_runtime_error():
    """A submission that crashes at runtime should be rejected."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('3\n', '9\n')]),
        completion='```python\nraise ValueError("crash")\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False
    assert len(result.tests) >= 1
    assert result.tests[0].passed is False

async def test_submit_cpp_compile_error():
    """A C++ submission that fails compilation should be rejected."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='cpp'),
        test_cases=_make_cases([('1\n', '1\n')]),
        completion='```cpp\nint main() { undefined_symbol(); }\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False

# ---------------------------------------------------------------------------
#  Timeout during evaluation
# ---------------------------------------------------------------------------

async def test_submit_python_timeout():
    """A submission that times out should be rejected."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python', run_timeout=0.3),
        test_cases=_make_cases([('1\n', '1\n')]),
        completion='```python\nimport time; time.sleep(10)\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False

# ---------------------------------------------------------------------------
#  Code extraction
# ---------------------------------------------------------------------------

async def test_submit_extracts_from_markdown_fence():
    """Code should be extracted from markdown fenced blocks."""
    completion = '''Here's my solution:

```python
import sys
n = int(sys.stdin.read())
print(n * n)
```

This works by squaring the input.'''
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('5\n', '25\n')]),
        completion=completion,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True
    assert 'n * n' in result.extracted_code
    assert "Here's my solution" not in result.extracted_code

async def test_submit_extracts_raw_code_no_fence():
    """If there are no fences, the extractor falls back to heuristics.
    Raw unfenced code that doesn't match heuristic patterns yields empty extracted_code."""
    completion = 'import sys\nn = int(sys.stdin.read())\nprint(n * n)'
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('5\n', '25\n')]),
        completion=completion,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert isinstance(result.accepted, bool)

async def test_submit_unfenced_code_via_incomplete_fence():
    """An incomplete fence (no closing ```) should still extract code."""
    completion = '```python\nimport sys\nn = int(sys.stdin.read())\nprint(n * n)'
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('5\n', '25\n')]),
        completion=completion,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True
    assert 'n * n' in result.extracted_code

async def test_submit_empty_completion():
    """An empty completion should be rejected, not crash."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('1\n', '1\n')]),
        completion='',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False

async def test_submit_garbage_completion():
    """A completion with no extractable code should be rejected."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('1\n', '1\n')]),
        completion='I cannot solve this problem. Here is a random sentence.',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False

# ---------------------------------------------------------------------------
#  Output comparison edge cases
# ---------------------------------------------------------------------------

async def test_submit_trailing_newline_tolerance():
    """Output with/without trailing newline should still match."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('1\n', '42\n')]),
        completion='```python\nprint(42)\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True

async def test_submit_trailing_whitespace_tolerance():
    """Trailing spaces on lines should be stripped during comparison."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('1\n', '42  \n')]),
        completion='```python\nprint(42)\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True

async def test_submit_case_insensitive_comparison():
    """Default comparison is case-insensitive (lower_cmp=True in check_stdio_test_case)."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('1\n', 'YES\n')]),
        completion='```python\nprint("yes")\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True

async def test_submit_multiline_output():
    """Multi-line output should match line by line."""
    code = '```python\nimport sys\nn=int(sys.stdin.read())\nfor i in range(1,n+1): print(i)\n```'
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('3\n', '1\n2\n3\n')]),
        completion=code,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True

async def test_submit_float_tolerance():
    """Floating-point outputs should match within relative tolerance."""
    code = '```python\nprint(3.141590001)\n```'
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('1\n', '3.14159\n')]),
        completion=code,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True

async def test_submit_float_too_far():
    """A floating-point output that is too far from expected should be rejected."""
    code = '```python\nprint(3.2)\n```'
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('1\n', '3.14159\n')]),
        completion=code,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False

async def test_submit_wrong_line_count():
    """Extra or missing output lines should cause rejection."""
    code = '```python\nprint(1)\nprint(2)\nprint(3)\n```'
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('1\n', '1\n2\n')]),
        completion=code,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False

# ---------------------------------------------------------------------------
#  run_all_cases flag
# ---------------------------------------------------------------------------

async def test_submit_run_all_cases_flag():
    """With run_all_cases=True, all test cases should be evaluated even after a failure."""
    cases = _make_cases([('1\n', '1\n'), ('2\n', '999\n'), ('3\n', '9\n')])  # 2nd case will fail
    code = '```python\nimport sys\nn=int(sys.stdin.read())\nprint(n*n)\n```'
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python', extra={'run_all_cases': True}),
        test_cases=cases,
        completion=code,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False
    assert len(result.tests) == 3
    assert result.tests[0].passed is True
    assert result.tests[1].passed is False
    assert result.tests[2].passed is True

async def test_submit_default_stops_on_first_failure():
    """Without run_all_cases, evaluation should stop after the first failed case."""
    cases = _make_cases([('1\n', '999\n'), ('2\n', '4\n'), ('3\n', '9\n')])  # 1st case wrong
    code = '```python\nimport sys\nn=int(sys.stdin.read())\nprint(n*n)\n```'
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=cases,
        completion=code,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is False
    assert any(not t.passed for t in result.tests)

# ---------------------------------------------------------------------------
#  Empty test cases
# ---------------------------------------------------------------------------

async def test_submit_with_empty_test_cases():
    """A submission with no test cases should result in accepted=True (vacuously)."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=[],
        completion='```python\nprint("anything")\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True
    assert len(result.tests) == 0

# ---------------------------------------------------------------------------
#  EvalResult structure
# ---------------------------------------------------------------------------

async def test_eval_result_has_extracted_code():
    """EvalResult should always include the extracted_code field."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('1\n', '1\n')]),
        completion='```python\nprint(1)\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.extracted_code is not None
    assert 'print(1)' in result.extracted_code

async def test_eval_result_tests_contain_exec_info():
    """Each EvalTestCase should have exec_info with run details."""
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python'),
        test_cases=_make_cases([('1\n', '1\n')]),
        completion='```python\nimport sys\nprint(sys.stdin.read().strip())\n```',
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert len(result.tests) == 1
    tc = result.tests[0]
    assert tc.exec_info is not None
    assert tc.exec_info.run_result is not None

# ---------------------------------------------------------------------------
#  Cross-language
# ---------------------------------------------------------------------------

async def test_submit_bash_via_inline_test_cases():
    """Bash submissions should work with inline test cases."""
    code = '```bash\nread line\necho "$line" | tr "[:lower:]" "[:upper:]"\n```'
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='bash'),
        test_cases=_make_cases([('hello\n', 'HELLO\n')]),
        completion=code,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True

# ---------------------------------------------------------------------------
#  Custom extract logic
# ---------------------------------------------------------------------------

async def test_submit_with_custom_extract_logic():
    """custom_extract_logic should override default extraction."""
    completion = '''SOLUTION_START
import sys
n = int(sys.stdin.read())
print(n * n)
SOLUTION_END'''
    custom_logic = '''
lines = completion.split('\\n')
start = None
end = None
for i, line in enumerate(lines):
    if 'SOLUTION_START' in line:
        start = i + 1
    if 'SOLUTION_END' in line:
        end = i
if start is not None and end is not None:
    code = '\\n'.join(lines[start:end])
    submit_code_blocks([CodeBlock(priority=40, code=code, language='python')])
'''
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python', custom_extract_logic=custom_logic),
        test_cases=_make_cases([('5\n', '25\n')]),
        completion=completion,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True
    assert 'n * n' in result.extracted_code

# ---------------------------------------------------------------------------
#  Stress: many test cases
# ---------------------------------------------------------------------------

async def test_submit_many_test_cases():
    """Submission against many test cases should work correctly."""
    cases = _make_cases([(f'{i}\n', f'{i*i}\n') for i in range(20)])
    code = '```python\nimport sys\nn=int(sys.stdin.read())\nprint(n*n)\n```'
    request = SubmitRequest(
        id=0,
        config=TestConfig(language='python', extra={'run_all_cases': True}),
        test_cases=cases,
        completion=code,
    )
    response = client.post('/submit', json=request.model_dump())
    assert response.status_code == 200
    result = EvalResult(**response.json())
    assert result.accepted is True
    assert len(result.tests) == 20
    assert all(t.passed for t in result.tests)
