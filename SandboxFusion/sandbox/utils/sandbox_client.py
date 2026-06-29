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

"""Internal sandbox API client for executing code and classifying results.

Provides functions to invoke the sandbox server's ``run_code()`` endpoint
directly (without HTTP), with configurable retry behavior via tenacity.
Also includes a result summarization system that maps ``RunCodeResponse``
statuses into customizable summary strings.
"""

import logging
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from sandbox.datasets.types import CommandRunStatus, RunCodeRequest, RunCodeResponse, RunStatus
from sandbox.server.sandbox_api import run_code

logger = logging.getLogger(__name__)


def on_retry_error(s):
    """Callback invoked when all retry attempts are exhausted.

    Logs the final error and raises an HTTP 500 exception with details about
    the failed sandbox request.

    Args:
        s: The tenacity ``RetryCallState`` containing the outcome and original
            arguments.

    Raises:
        HTTPException: Always raised with status code 500.
    """
    e = s.outcome.exception()
    logger.error(f'give up requesting sandbox. error: {e}. request: {s.args[0].model_dump_json(indent=2)}')
    raise HTTPException(status_code=500, detail=f'failed to request sandbox: {e}')


def before_retry_sleep(s):
    """Callback invoked before sleeping between retry attempts.

    Logs a warning with the current attempt number, the error, and the
    request payload.

    Args:
        s: The tenacity ``RetryCallState`` containing attempt metadata.
    """
    logger.warning(
        f'error requesting sandbox for {s.attempt_number} time(s), will retry... error: {s.outcome.exception()}. request: {s.args[0].model_dump_json(indent=2)}'
    )


@retry(wait=wait_exponential_jitter(),
       stop=stop_after_attempt(1),
       before_sleep=before_retry_sleep,
       retry_error_callback=on_retry_error)
async def run_code_in_sandbox(request: RunCodeRequest) -> RunCodeResponse:
    """Execute code in the sandbox with no retries (1 attempt).

    Calls the server's ``run_code()`` function directly (in-process, no HTTP).
    Raises an exception if the sandbox returns a ``SandboxError`` status, which
    triggers the retry/error callback mechanism.

    Args:
        request: The code execution request specifying language, code, stdin,
            and timeout parameters.

    Returns:
        The ``RunCodeResponse`` from the sandbox execution.

    Raises:
        Exception: If the sandbox response status is ``SandboxError``.
    """
    resp = await run_code(request)
    if resp.status == RunStatus.SandboxError:
        raise Exception(f'Sandbox responded with error: {resp.message}')
    return resp


@retry(wait=wait_exponential_jitter(),
       stop=stop_after_attempt(5),
       before_sleep=before_retry_sleep,
       retry_error_callback=on_retry_error)
async def run_code_in_sandbox_w_retry(request: RunCodeRequest) -> RunCodeResponse:
    """Execute code in the sandbox with up to 5 retry attempts.

    Same as :func:`run_code_in_sandbox` but configured with exponential
    backoff jitter and up to 5 attempts before giving up.

    Args:
        request: The code execution request specifying language, code, stdin,
            and timeout parameters.

    Returns:
        The ``RunCodeResponse`` from the sandbox execution.

    Raises:
        Exception: If the sandbox response status is ``SandboxError``
            (triggers retry).
    """
    resp = await run_code(request)
    if resp.status == RunStatus.SandboxError:
        raise Exception(f'Sandbox responded with error: {resp.message}')
    return resp


class SummaryMapping(BaseModel):
    """Configuration for mapping ``RunCodeResponse`` statuses to custom summary strings.

    Allows callers to define custom status labels for each possible outcome of
    a sandbox execution. Fields set to ``None`` fall back to the ``Failed``
    status string.

    Attributes:
        Success: Status string for successful execution. Defaults to
            ``RunStatus.Success``.
        Failed: Status string for general failure. Defaults to
            ``RunStatus.Failed``.
        CompileFailed: Optional status string for compilation failure.
            Falls back to ``Failed`` if ``None``.
        CompileTimeout: Optional status string for compilation timeout.
            Falls back to ``Failed`` if ``None``.
        RunFailed: Optional status string for runtime failure.
            Falls back to ``Failed`` if ``None``.
        RunTimeout: Optional status string for runtime timeout.
            Falls back to ``Failed`` if ``None``.
    """

    Success: str = RunStatus.Success
    Failed: str = RunStatus.Failed
    CompileFailed: Optional[str] = None
    CompileTimeout: Optional[str] = None
    RunFailed: Optional[str] = None
    RunTimeout: Optional[str] = None


def summary_result(result: RunCodeResponse, mapping: SummaryMapping) -> str:
    """Classify a ``RunCodeResponse`` into a summary status string.

    Examines the compile and run results to determine the appropriate status
    label according to the provided ``SummaryMapping``. Handles the following
    cases:

    - No compile/run results: uses ``Success`` or ``Failed`` based on overall status.
    - Compile-only result: checks for timeout or non-zero return code.
    - Run result present: checks for timeout or non-zero return code.
    - Non-zero return codes map to ``CompileFailed``/``RunFailed``.
    - Timeouts map to ``CompileTimeout``/``RunTimeout``.

    Args:
        result: The sandbox execution response to classify.
        mapping: The status string mapping configuration.

    Returns:
        The summary status string from the mapping.

    Raises:
        Exception: If the result is in an unexpected or invalid state.
    """
    if result.compile_result is None and result.run_result is None:
        if result.status == RunStatus.Success:
            return mapping.Success
        if result.status == RunStatus.Failed:
            return mapping.Failed
        raise Exception(f'unexpected result status {result.status}')
    if result.run_result is None:
        if result.compile_result.status == CommandRunStatus.TimeLimitExceeded:
            return mapping.CompileTimeout or mapping.Failed
        return_code = result.compile_result.return_code
        if return_code is None:
            raise Exception(f'invalid sandbox result: no return code with status {result.compile_result.status}')
        if return_code != 0:
            return mapping.CompileFailed or mapping.Failed
        raise Exception(f'invalid sandbox result: compiled successfully with no run result')
    if result.run_result.status == CommandRunStatus.TimeLimitExceeded:
        return mapping.RunTimeout or mapping.Failed
    return_code = result.run_result.return_code
    if return_code is None:
        raise Exception(f'invalid sandbox result: no return code with status {result.run_result.status}')
    if return_code != 0:
        return mapping.RunFailed or mapping.Failed
    return mapping.Success
