# Python SDK

SandboxFusion provides a Python client SDK (`sandbox-fusion`) for programmatic access to the sandbox API. It supports both synchronous and asynchronous interfaces, automatic retries, concurrent batch execution, and configurable timeouts.

## Installation

Requires Python >= 3.8. Install from the local source included in this repository (do **not** `pip install sandbox-fusion` from PyPI — that installs the upstream Bytedance package, not this fork):

```bash
pip install ./scripts/client
```

The SDK source is located in `scripts/client/src/sandbox_fusion/`.

## Configuring the API Endpoint

By default, the SDK uses the `SANDBOX_FUSION_ENDPOINT` environment variable. If unset, it defaults to `http://localhost:8000`.

### Option 1: Environment variable

```bash
export SANDBOX_FUSION_ENDPOINT="http://localhost:8080"
```

### Option 2: `set_endpoint()` function

```python
from sandbox_fusion import set_endpoint

set_endpoint("http://localhost:8080")
```

### Option 3: Per-call `endpoint` parameter

```python
from sandbox_fusion import run_code, RunCodeRequest

run_code(
    RunCodeRequest(code='print(123)', language='python'),
    endpoint="http://localhost:8080"
)
```

The per-call `endpoint` parameter overrides the global setting.

## API Functions

All HTTP API endpoints have a corresponding SDK function. Each function accepts a pydantic request model and returns a pydantic response model.

### run_code

Execute code in any supported language.

```python
from sandbox_fusion import run_code, RunCodeRequest

# Default retry count: 5 attempts
result = run_code(RunCodeRequest(
    code='print(123)',
    language='python'
))
print(result.run_result.stdout)  # "123\n"

# Custom retry count
result = run_code(
    RunCodeRequest(code='print(123)', language='python'),
    max_attempts=10
)
```

- **Request:** `RunCodeRequest`
- **Response:** `RunCodeResponse`
- **Default retries:** 5

### submit

Submit a completion for evaluation against inline test cases.

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
print(result.accepted)  # True or False
```

- **Request:** `SubmitRequest`
- **Response:** `EvalResult`
- **Default retries:** 5

### submit_safe

Same as `submit`, but returns a rejected `EvalResult` instead of raising an exception on failure.

```python
from sandbox_fusion import submit_safe, SubmitRequest

result = submit_safe(SubmitRequest(
    id='problem-1',
    completion='some code',
    config={'language': 'python'},
    test_cases=[
        {'input': {'stdin': '1\n'}, 'output': {'stdout': '1\n'}},
    ]
))
# Always returns an EvalResult, never raises
print(result.accepted)
```

- **Request:** `SubmitRequest`
- **Response:** `EvalResult`
- **Default retries:** 5

## Asynchronous Interface

All functions have async versions, imported by appending `_async` to the function name:

```python
import asyncio
from sandbox_fusion import run_code_async, RunCodeRequest

async def main():
    result = await run_code_async(RunCodeRequest(
        code='print("hello")',
        language='python'
    ))
    print(result.run_result.stdout)

asyncio.run(main())
```

Available async functions:
- `run_code_async`
- `submit_async`
- `submit_safe_async`

## Concurrent Requests

The SDK provides `run_concurrent` for batch-executing operations in parallel:

```python
from sandbox_fusion import set_endpoint, run_concurrent, run_code, RunCodeRequest

set_endpoint('http://localhost:8080')

codes = [f'print({i})' for i in range(100, 200)]
results = run_concurrent(
    run_code,
    args=[[RunCodeRequest(code=c, language='python')] for c in codes]
)

for r in results:
    print(r.run_result.stdout.strip())
```

`run_concurrent` accepts any SDK function and a list of argument lists. It executes all calls concurrently and returns results in order.

## Timeout Settings

Time-consuming functions (`submit`, `run_code`) support a `client_timeout` parameter that controls how long the SDK waits for a response before timing out:

```python
from sandbox_fusion import run_code, RunCodeRequest

# Will timeout after 3 seconds on the client side
result = run_code(
    RunCodeRequest(
        code='import time; time.sleep(4); print(123)',
        language='python'
    ),
    max_attempts=1,
    client_timeout=3
)
```

This is a client-side timeout (how long the HTTP request waits). It is separate from the server-side `run_timeout` in the request body (how long the sandbox allows the code to execute).

## Complete Example: RL Training Loop

```python
from sandbox_fusion import set_endpoint, submit, SubmitRequest, run_concurrent

set_endpoint('http://localhost:8080')

# Suppose you have a list of (problem_id, completion, test_cases) from your LLM
problems = [
    {
        'id': 'add-two-numbers',
        'completion': '```python\na,b=map(int,input().split())\nprint(a+b)\n```',
        'test_cases': [
            {'input': {'stdin': '1 2\n'}, 'output': {'stdout': '3\n'}},
            {'input': {'stdin': '5 7\n'}, 'output': {'stdout': '12\n'}},
        ]
    },
    # ... more problems
]

# Build submit requests
requests = [
    SubmitRequest(
        id=p['id'],
        completion=p['completion'],
        config={'language': 'python'},
        test_cases=p['test_cases']
    )
    for p in problems
]

# Evaluate all concurrently
results = run_concurrent(submit, args=[[r] for r in requests])

# Use accepted as reward signal
rewards = [1.0 if r.accepted else 0.0 for r in results]
print(f"Pass rate: {sum(rewards) / len(rewards):.1%}")
```

## Request and Response Models

The SDK re-exports the same pydantic models used by the server:

| Model | Description |
|-------|-------------|
| `RunCodeRequest` | Request for `/run_code` |
| `RunCodeResponse` | Response from `/run_code` |
| `SubmitRequest` | Request for `/submit` |
| `EvalResult` | Response from `/submit` |
| `TestConfig` | Evaluation config (language, timeouts, etc.) |
| `GeneralStdioTest` | A single stdin/stdout test case |
| `CommandRunResult` | Result of a single command execution |

See [API Reference](api-reference.md) for full field documentation.
