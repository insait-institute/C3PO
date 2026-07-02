# API Reference

SandboxFusion exposes four HTTP endpoints. The server runs on FastAPI and listens on port 8080 by default.

## Endpoints Overview

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Redirects to the documentation site |
| `GET` | `/v1/ping` | Health check |
| `POST` | `/run_code` | Execute code in any supported language |
| `POST` | `/submit` | Evaluate a completion against inline test cases |

---

## GET /

Redirects to the static documentation site. Not intended for programmatic use.

---

## GET /v1/ping

Health check endpoint. Returns the string `"pong"` with a 200 status code when the server is running.

**Example:**

```bash
curl http://localhost:8080/v1/ping
```

**Response:**

```
"pong"
```

---

## POST /run_code

Execute source code inside a sandboxed environment and return the results.

### Request Body (`RunCodeRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `code` | `string` | Yes | -- | Source code to execute |
| `language` | `string` | Yes | -- | Language identifier (see [Supported Languages](getting-started.md#supported-languages)) |
| `compile_timeout` | `float` | No | `10` | Maximum seconds for compilation (compiled languages only) |
| `run_timeout` | `float` | No | `10` | Maximum seconds for execution |
| `memory_limit_MB` | `int` | No | `-1` | Memory limit in MB. `-1` uses the server default (`sandbox.default_memory_limit_mb`, typically 8192 MB). |
| `stdin` | `string` | No | `null` | String to pipe into the program's standard input |
| `files` | `Dict[string, string]` | No | `{}` | File path to base64-encoded content mapping; files are written into the sandbox before execution |
| `fetch_files` | `List[string]` | No | `[]` | File paths to read back from the sandbox after execution |

### Response Body (`RunCodeResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `status` | `RunStatus` | `"Success"`, `"Failed"`, or `"SandboxError"` |
| `message` | `string` | Error details (empty on success) |
| `compile_result` | `CommandRunResult` or `null` | Compilation output (null for interpreted languages) |
| `run_result` | `CommandRunResult` or `null` | Execution output |
| `executor_pod_name` | `string` or `null` | Kubernetes pod name (if running in k8s) |
| `files` | `Dict[string, string]` | Requested files as base64-encoded content |

### RunStatus Values

| Value | Meaning |
|-------|---------|
| `Success` | All commands (compile + run) finished with exit code 0 |
| `Failed` | The user's code failed (non-zero exit code or time limit exceeded) |
| `SandboxError` | Infrastructure/sandbox-level error (runner exception, internal failure) |

### CommandRunResult

| Field | Type | Description |
|-------|------|-------------|
| `status` | `CommandRunStatus` | `"Finished"`, `"Error"`, or `"TimeLimitExceeded"` |
| `execution_time` | `float` or `null` | Wall-clock seconds elapsed |
| `return_code` | `int` or `null` | Process exit code |
| `stdout` | `string` or `null` | Captured standard output |
| `stderr` | `string` or `null` | Captured standard error |

### Examples

**Simple Python execution:**

```bash
curl -X POST http://localhost:8080/run_code \
  -H "Content-Type: application/json" \
  -d '{"code": "print(\"Hello, world!\")", "language": "python"}'
```

```json
{
  "status": "Success",
  "message": "",
  "compile_result": null,
  "run_result": {
    "status": "Finished",
    "execution_time": 0.016,
    "return_code": 0,
    "stdout": "Hello, world!\n",
    "stderr": ""
  },
  "executor_pod_name": null,
  "files": {}
}
```

**Compiled language (C++):**

```bash
curl -X POST http://localhost:8080/run_code \
  -H "Content-Type: application/json" \
  -d '{
    "code": "#include <iostream>\nint main() { std::cout << \"Hello\" << std::endl; return 0; }",
    "language": "cpp",
    "compile_timeout": 30,
    "run_timeout": 10
  }'
```

```json
{
  "status": "Success",
  "message": "",
  "compile_result": {
    "status": "Finished",
    "execution_time": 0.458,
    "return_code": 0,
    "stdout": "",
    "stderr": ""
  },
  "run_result": {
    "status": "Finished",
    "execution_time": 0.002,
    "return_code": 0,
    "stdout": "Hello\n",
    "stderr": ""
  },
  "executor_pod_name": null,
  "files": {}
}
```

**With stdin:**

```bash
curl -X POST http://localhost:8080/run_code \
  -H "Content-Type: application/json" \
  -d '{
    "code": "x = input()\nprint(f\"You said: {x}\")",
    "language": "python",
    "stdin": "hello\n"
  }'
```

**With file upload and download:**

```bash
curl -X POST http://localhost:8080/run_code \
  -H "Content-Type: application/json" \
  -d '{
    "code": "data = open(\"input.txt\").read()\nopen(\"output.txt\", \"w\").write(data.upper())\nprint(\"done\")",
    "language": "python",
    "files": {"input.txt": "aGVsbG8gd29ybGQ="},
    "fetch_files": ["output.txt"]
  }'
```

The `files` field maps file paths to **base64-encoded** content. Files are written into the sandbox working directory (a temp directory under `/tmp`) before code execution. Relative paths are resolved relative to this temp directory; absolute paths are also supported.

The `fetch_files` field lists paths to retrieve after execution. Their contents are returned base64-encoded in the response `files` field.

**Python using requests:**

```python
import json
import requests

response = requests.post('http://localhost:8080/run_code', json={
    'code': '''
#include <iostream>

int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
''',
    'language': 'cpp',
})

print(json.dumps(response.json(), indent=2))
```

---

## POST /submit

Evaluate an LLM-generated completion against a set of inline stdin/stdout test cases. The sandbox extracts code from the completion (handling fenced code blocks, heuristics), runs it against each test case, and reports pass/fail.

### Request Body (`SubmitRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `id` | `int` or `string` | Yes | -- | Problem identifier |
| `completion` | `string` | Yes | -- | Raw LLM output (code is extracted automatically) |
| `config` | `TestConfig` | Yes | -- | Evaluation configuration |
| `test_cases` | `List[GeneralStdioTest]` | Yes | -- | Inline stdin/stdout test cases |

### TestConfig

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `language` | `string` | No | `null` | Language identifier (auto-detected if null) |
| `locale` | `string` | No | `null` | Locale hint (e.g. `"en"`, `"zh"`) |
| `compile_timeout` | `float` | No | `null` | Max compilation seconds (uses default if null) |
| `run_timeout` | `float` | No | `null` | Max execution seconds (uses default if null) |
| `custom_extract_logic` | `string` | No | `null` | Custom Python code for code extraction (advanced) |
| `extra` | `Dict` | No | `{}` | Catch-all for additional configuration |

### GeneralStdioTest

| Field | Type | Description |
|-------|------|-------------|
| `input` | `Dict[string, string]` | Input streams, typically `{"stdin": "..."}` |
| `output` | `Dict[string, string]` | Expected output, typically `{"stdout": "..."}` |

### Response Body (`EvalResult`)

| Field | Type | Description |
|-------|------|-------------|
| `id` | `int` or `string` | Problem identifier |
| `accepted` | `bool` | `true` if ALL test cases passed |
| `extracted_code` | `string` | Code extracted from the completion |
| `full_code` | `string` or `null` | Complete code that was actually compiled/run (may include harness) |
| `test_code` | `string` or `null` | Test/driver code used for evaluation |
| `tests` | `List[EvalTestCase]` | Per-test-case results |
| `extracted_type` | `string` or `null` | Extraction method: `"fenced"`, `"incomplete_fenced"`, `"heuristic"`, or `"empty"` |
| `extra` | `Dict` or `null` | Optional extra metadata |

### EvalTestCase

| Field | Type | Description |
|-------|------|-------------|
| `passed` | `bool` | Whether this test case passed |
| `exec_info` | `RunCodeResponse` | Full execution details (same schema as /run_code response) |
| `test_info` | `Dict` or `null` | Optional extra judge information |

### Code Extraction

The `completion` field accepts raw LLM output. SandboxFusion automatically extracts code using `default_extract_helper()`, which applies the following precedence:

1. **Fenced code blocks** (priority 30): `` ```python ... ``` `` or `` ```lang ... ``` ``
2. **Incomplete fenced blocks** (priority 20): Opening fence without closing
3. **Heuristic extraction** (priority 10): Detects code-like content without fences

You can override this with `custom_extract_logic` in `TestConfig`, which receives a `List[CodeBlock]` and calls `submit_code_blocks(cbs)` with custom priorities (typically priority 40 to take precedence over built-in extractors).

### Examples

**Basic evaluation:**

```bash
curl -X POST http://localhost:8080/submit \
  -H "Content-Type: application/json" \
  -d '{
    "id": "problem-1",
    "completion": "```python\na, b = map(int, input().split())\nprint(a + b)\n```",
    "config": {"language": "python"},
    "test_cases": [
      {"input": {"stdin": "1 2\n"}, "output": {"stdout": "3\n"}},
      {"input": {"stdin": "10 20\n"}, "output": {"stdout": "30\n"}}
    ]
  }'
```

**Response:**

```json
{
  "id": "problem-1",
  "accepted": true,
  "extracted_code": "a, b = map(int, input().split())\nprint(a + b)",
  "full_code": "a, b = map(int, input().split())\nprint(a + b)",
  "test_code": null,
  "tests": [
    {
      "passed": true,
      "exec_info": {
        "status": "Success",
        "message": "",
        "compile_result": null,
        "run_result": {
          "status": "Finished",
          "execution_time": 0.017,
          "return_code": 0,
          "stdout": "3\n",
          "stderr": ""
        },
        "executor_pod_name": null,
        "files": {}
      },
      "test_info": null
    },
    {
      "passed": true,
      "exec_info": {
        "status": "Success",
        "message": "",
        "compile_result": null,
        "run_result": {
          "status": "Finished",
          "execution_time": 0.015,
          "return_code": 0,
          "stdout": "30\n",
          "stderr": ""
        },
        "executor_pod_name": null,
        "files": {}
      },
      "test_info": null
    }
  ],
  "extracted_type": "fenced",
  "extra": null
}
```

**Using the Python SDK:**

```python
from sandbox_fusion import submit, SubmitRequest

result = submit(SubmitRequest(
    id='problem-1',
    completion='```python\na, b = map(int, input().split())\nprint(a + b)\n```',
    config={'language': 'python'},
    test_cases=[
        {'input': {'stdin': '1 2\n'}, 'output': {'stdout': '3\n'}},
        {'input': {'stdin': '10 20\n'}, 'output': {'stdout': '30\n'}},
    ]
))
print(result.accepted)  # True
```

### RL Training Loop Pattern

The `/submit` endpoint is designed for use as a reward signal in RL training loops:

1. Prepare problems with stdin/stdout test cases (from any source).
2. Generate completions by feeding prompts to the LLM being trained.
3. Submit each completion via `POST /submit` with inline test cases.
4. The `accepted` field in `EvalResult` serves as the binary reward signal (all tests passed = reward 1, otherwise 0).
5. Per-test-case results in `tests` provide fine-grained feedback for partial-credit schemes.
