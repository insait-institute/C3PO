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

"""Sandbox code-execution API.

Defines the ``POST /run_code`` endpoint that accepts source code in a
specified language, dispatches it to the appropriate language runner from
:mod:`sandbox.runners`, and returns a structured response containing
compilation and execution results along with an overall status
(Success, Failed, or SandboxError).
"""

import asyncio
import os
import traceback
from enum import Enum
from typing import Dict, List, Optional, Tuple

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from sandbox.configs.run_config import RunConfig
from sandbox.runners import (
    CODE_RUNNERS,
    CodeRunArgs,
    CodeRunResult,
    CommandRunResult,
    CommandRunStatus,
    Language,
)

sandbox_router = APIRouter()
logger = structlog.stdlib.get_logger()

_sandbox_config = RunConfig.get_instance_sync().sandbox
_run_code_semaphore: Optional[asyncio.Semaphore] = None


def _get_run_code_semaphore() -> Optional[asyncio.Semaphore]:
    """Lazily create the concurrency semaphore on first use.

    Deferred to first request so the semaphore is bound to the running
    event loop rather than created at module-import time.
    """
    global _run_code_semaphore
    if _run_code_semaphore is None and _sandbox_config.max_concurrency > 0:
        _run_code_semaphore = asyncio.Semaphore(_sandbox_config.max_concurrency)
    return _run_code_semaphore


async def _acquire_slot(sem: Optional[asyncio.Semaphore], queue_timeout: float) -> bool:
    """Acquire an execution slot, applying admission control.

    Returns ``True`` when a slot was acquired (the caller must release it),
    ``False`` when there is no semaphore (concurrency limiting disabled, so
    nothing to release).

    When ``queue_timeout > 0`` and no slot becomes available within that many
    seconds, raises :class:`fastapi.HTTPException` with status ``503`` so an
    over-subscribed server fails fast and retryably instead of holding the
    connection until the client read-times-out.  ``queue_timeout <= 0`` keeps
    the original behaviour of waiting indefinitely for a slot.
    """
    if sem is None:
        return False
    if queue_timeout and queue_timeout > 0:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=queue_timeout)
        except asyncio.TimeoutError:
            logger.warning(f'admission timeout: no slot within {queue_timeout}s; rejecting with 503')
            raise HTTPException(status_code=503, detail='sandbox queue saturated')
    else:
        await sem.acquire()
    return True


class RunCodeRequest(BaseModel):
    """Request body for the ``POST /run_code`` endpoint.

    Attributes:
        compile_timeout: Maximum seconds allowed for the compilation step
            (applicable to compiled languages only).  Defaults to 10.
        run_timeout: Maximum seconds allowed for code execution.
            Defaults to 10.
        memory_limit_MB: Hard memory cap in megabytes for the sandbox
            process.  A value of ``-1`` (the default) uses the config
            default (``sandbox.default_memory_limit_mb``, typically 8192 MB).
        code: Source code to execute.
        stdin: Optional string piped to the process's standard input.
        language: Target language or execution mode (must be a key in
            :data:`sandbox.runners.CODE_RUNNERS`).
        files: Mapping of file paths to base64-encoded contents that
            will be materialised inside the sandbox before execution.
        fetch_files: List of file paths to read back from the sandbox
            after execution and return as base64 in the response.
    """

    compile_timeout: float = Field(10, description='compile timeout for compiled languages')
    run_timeout: float = Field(10, description='code run timeout')
    memory_limit_MB: int = Field(-1, description='maximum memory allowed in megabytes')
    code: str = Field(..., examples=['print("hello")'], description='the code to run')
    stdin: Optional[str] = Field(None, examples=[''], description='optional string to pass into stdin')
    language: Language = Field(..., examples=['python'], description='the language or execution mode to run the code')
    files: Dict[str, Optional[str]] = Field({}, description='a dict from file path to base64 encoded file content')
    fetch_files: List[str] = Field([], description='a list of file paths to fetch after code execution')


class RunStatus(str, Enum):
    """High-level outcome of a code-execution request.

    Members:
        Success: All commands (compile and run) finished with a zero
            exit code and no infrastructure errors.
        Failed: The user's code itself failed -- either a non-zero exit
            code or a time-limit-exceeded condition.
        SandboxError: An infrastructure or sandbox-level error occurred
            (e.g. the runner raised an exception or reported an Error
            status).
    """

    # all command finished successfully
    Success = 'Success'
    # one of the process has non-zero return code
    Failed = 'Failed'
    # error on sandbox side
    SandboxError = 'SandboxError'


class RunCodeResponse(BaseModel):
    """Response body returned by the ``POST /run_code`` endpoint.

    Attributes:
        status: Aggregate outcome of the execution request.
        message: Human-readable detail when ``status`` is not
            :attr:`RunStatus.Success` (empty string on success).
        compile_result: Output of the compilation step, if applicable.
        run_result: Output of the execution step.
        executor_pod_name: Kubernetes pod name that handled the request
            (populated from the ``MY_POD_NAME`` environment variable).
        files: Mapping of requested file paths to their base64-encoded
            contents, as specified by
            :attr:`RunCodeRequest.fetch_files`.
    """

    status: RunStatus
    message: str
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    executor_pod_name: Optional[str] = None
    files: Dict[str, str] = {}


def parse_run_status(result: CodeRunResult) -> Tuple[RunStatus, str]:
    """Interpret a :class:`CodeRunResult` into a high-level status and message.

    Inspects both the compile and run results (if present) in order and
    applies the following precedence rules:

    1. If any step has :attr:`CommandRunStatus.Error`, return
       :attr:`RunStatus.SandboxError` with that step's stderr.
    2. If any step hit a time-limit exceeded condition, return
       :attr:`RunStatus.Failed`.
    3. If any step exited with a non-zero return code, return
       :attr:`RunStatus.Failed`.
    4. Otherwise return :attr:`RunStatus.Success`.

    Args:
        result: The raw result produced by a language runner.

    Returns:
        A ``(RunStatus, message)`` tuple.  The message is non-empty only
        for :attr:`RunStatus.SandboxError`.
    """
    outcomes = []
    retcodes = []
    err_msgs = []
    if result.compile_result is not None:
        outcomes.append(result.compile_result.status)
        err_msgs.append(result.compile_result.stderr or '')
        if result.compile_result.return_code is not None:
            retcodes.append(result.compile_result.return_code)
    if result.run_result is not None:
        outcomes.append(result.run_result.status)
        err_msgs.append(result.run_result.stderr or '')
        if result.run_result.return_code is not None:
            retcodes.append(result.run_result.return_code)

    for o, m in zip(outcomes, err_msgs):
        if o == CommandRunStatus.Error:
            return RunStatus.SandboxError, m
    if any([o == CommandRunStatus.TimeLimitExceeded for o in outcomes]):
        return RunStatus.Failed, ''
    if any([r != 0 for r in retcodes]):
        return RunStatus.Failed, ''
    # no error, no tle and no non-zero return codes -> success
    return RunStatus.Success, ''


@sandbox_router.post("/run_code", response_model=RunCodeResponse, tags=['sandbox'])
async def run_code(request: RunCodeRequest):
    """Execute arbitrary source code inside a sandboxed environment.

    Dispatches the request to the runner registered in
    :data:`sandbox.runners.CODE_RUNNERS` for the requested language,
    translates the runner's :class:`CodeRunResult` into a
    :class:`RunCodeResponse`, and catches any unexpected exceptions as a
    :attr:`RunStatus.SandboxError`.

    Concurrent executions are bounded by ``sandbox.max_concurrency`` from
    the YAML configuration.

    Args:
        request: Validated request payload.

    Returns:
        A :class:`RunCodeResponse` containing compilation/execution
        output and an overall status.
    """
    resp = RunCodeResponse(status=RunStatus.Success, message='', executor_pod_name=os.environ.get('MY_POD_NAME'))

    # Admission control: when the server is at capacity, a request waits for a
    # `max_concurrency` slot.  Without a bound it waits until the client's read
    # timeout fires (presenting as a client-side read timeout that the trainer
    # then retries with a long, reward-step-stalling backoff).  With
    # `queue_timeout` set, `_acquire_slot` gives up after that budget and raises
    # a 503 instead, so over-subscription degrades gracefully.  Done outside the
    # try/except below so the 503 surfaces as a real HTTP status rather than
    # being swallowed into a 200 SandboxError.
    sem = _get_run_code_semaphore()
    acquired = await _acquire_slot(sem, _sandbox_config.queue_timeout)

    try:
        logger.debug(
            f'start processing {request.language} request with code ```\n{request.code[:100]}\n``` and files {list(request.files.keys())}...(memory_limit: {request.memory_limit_MB}MB)'
        )
        result = await CODE_RUNNERS[request.language](CodeRunArgs(**request.model_dump()))

        resp.compile_result = result.compile_result
        resp.run_result = result.run_result
        resp.files = result.files
        resp.status, message = parse_run_status(result)
        if resp.status == RunStatus.SandboxError:
            resp.message = message
    except Exception as e:
        message = f'exception on running code {request.code}: {e} {traceback.print_tb(e.__traceback__)}'
        logger.warning(message)
        resp.message = message
        resp.status = RunStatus.SandboxError
    finally:
        if acquired:
            sem.release()

    return resp


