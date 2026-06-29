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

"""Pydantic data models mirroring the SandboxFusion server types.

All models use ``pydantic.v1`` for backward compatibility with both Pydantic
v1 and v2 installations. When Pydantic v2 is installed, the ``pydantic.v1``
compatibility shim is used; otherwise the original ``pydantic`` module is
imported directly.

The models are split into two groups:

**Sandbox models** -- represent code-execution requests and responses:
    :class:`CommandRunStatus`, :class:`CommandRunResult`, :class:`CodeRunArgs`,
    :class:`CodeRunResult`, :class:`RunStatus`, :class:`RunCodeRequest`,
    :class:`RunCodeResponse`, :class:`SummaryMapping`

**Eval models** -- represent evaluation/submission payloads:
    :class:`GeneralStdioTest`, :class:`TestConfig`, :class:`EvalTestCase`,
    :class:`EvalResult`, :class:`SubmitRequest`
"""

from enum import Enum
from typing import Dict, Literal, Optional, List, Any, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from pydantic.v1 import BaseModel, Field
else:
    try:
        from pydantic.v1 import BaseModel, Field
    except ImportError:
        from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Sandbox-related models
# ---------------------------------------------------------------------------


class CommandRunStatus(str, Enum):
    """Terminal status of a single command (compile or run) execution."""
    Finished = 'Finished'
    Error = 'Error'
    TimeLimitExceeded = 'TimeLimitExceeded'


class CommandRunResult(BaseModel):
    """Result of executing a single command (compilation step or run step).

    Contains the exit status, optional timing, return code, and captured
    stdout/stderr output.
    """
    status: CommandRunStatus
    execution_time: Optional[float] = None
    return_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class CodeRunArgs(BaseModel):
    """Arguments for a code execution request (internal representation).

    Holds the source code, optional supplementary files, timeout settings,
    optional stdin input, and a list of file paths to retrieve after execution.
    """
    code: str
    files: Dict[str, str] = {}
    compile_timeout: float = 10
    run_timeout: float = 10
    stdin: Optional[str] = None
    fetch_files: List[str] = []


class CodeRunResult(BaseModel):
    """Internal result of a code execution, pairing compile and run outcomes."""
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    files: Dict[str, str] = {}


#: Union of all supported language identifiers accepted by the sandbox server.
Language = Literal['python', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'php', 'csharp', 'bash', 'typescript', 'sql',
                   'rust', 'lua', 'R', 'perl', 'D_ut', 'ruby', 'scala', 'julia', 'pytest', 'junit',
                   'kotlin_script', 'jest', 'verilog', 'lean', 'swift', 'racket']


class RunStatus(str, Enum):
    """High-level outcome status of a ``/run_code`` request."""
    Success = 'Success'
    Failed = 'Failed'
    SandboxError = 'SandboxError'


class RunCodeRequest(BaseModel):
    """Request payload for the ``/run_code`` API endpoint.

    Specifies the code to execute, the target language, timeout/memory limits,
    optional stdin, supplementary files, and files to retrieve post-execution.
    """
    compile_timeout: float = Field(10, description='compile timeout for compiled languages')
    run_timeout: float = Field(10, description='code run timeout')
    memory_limit_MB: int = Field(-1, description='maximum memory allowed in megabytes')
    code: str = Field(..., examples=['print("hello")'], description='the code to run')
    stdin: Optional[str] = Field(None, examples=[''], description='optional string to pass into stdin')
    language: Language = Field(..., examples=['python'], description='the language or execution mode to run the code')
    files: Dict[str, Optional[str]] = Field({}, description='a dict from file path to base64 encoded file content')
    fetch_files: List[str] = Field([], description='a list of file paths to fetch after code execution')


class RunCodeResponse(BaseModel):
    """Response payload from the ``/run_code`` API endpoint.

    Contains the overall status, an optional human-readable message, the
    compile and run sub-results, and any fetched files.
    """
    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}


class SummaryMapping(BaseModel):
    """Configurable mapping from execution outcomes to summary status strings.

    Used by :func:`sandbox_fusion.client.summary_run_code_result` to translate
    a :class:`RunCodeResponse` into a single descriptive string. Fields that
    are ``None`` fall back to the generic ``Failed`` string.
    """
    Success: str = RunStatus.Success
    Failed: str = RunStatus.Failed
    CompileFailed: Optional[str] = None
    CompileTimeout: Optional[str] = None
    RunFailed: Optional[str] = None
    RunTimeout: Optional[str] = None


# ---------------------------------------------------------------------------
# Eval-related models
# ---------------------------------------------------------------------------


class GeneralStdioTest(BaseModel):
    """A single stdin/stdout test case for evaluation.

    ``input`` and ``output`` are dicts mapping named streams to their content.
    """
    input: Dict[str, str]
    output: Dict[str, str]


class TestConfig(BaseModel):
    """Configuration for an evaluation submission.

    Specifies the target language, locale, timeout overrides, an optional
    custom extraction logic string, and an arbitrary ``extra`` dict for
    dataset-specific settings.
    """
    __test__ = False
    language: Optional[Language] = None
    locale: Optional[str] = None
    compile_timeout: Optional[float] = None
    run_timeout: Optional[float] = None
    custom_extract_logic: Optional[str] = None
    extra: Dict[str, Any] = {}


class EvalTestCase(BaseModel):
    """Result of a single test case within an evaluation.

    Records whether the test passed, the full execution info, and optional
    test-specific metadata.
    """
    passed: bool
    exec_info: RunCodeResponse
    test_info: Optional[Dict[str, Any]] = None


class EvalResult(BaseModel):
    """Overall result of a ``/submit`` evaluation request.

    Indicates whether the submission was accepted (all tests passed), the
    extracted code, and per-test-case details.
    """
    id: Union[int, str]
    accepted: bool
    extracted_code: str
    full_code: Optional[str] = None
    test_code: Optional[str] = None
    tests: List[EvalTestCase]
    extracted_type: Optional[Literal['fenced', 'incomplete_fenced', 'heuristic', 'empty']] = None
    extra: Optional[Dict] = None


class SubmitRequest(BaseModel):
    """Request payload for the ``/submit`` API endpoint.

    Contains the submission ID, the raw completion text from which code will
    be extracted, the test configuration, and the list of test cases.
    """
    id: Union[int, str]
    completion: str
    config: TestConfig
    test_cases: List[GeneralStdioTest]
