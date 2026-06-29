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

"""Synchronous HTTP client for the SandboxFusion server.

Provides functions for sending code-execution and evaluation-submission
requests to a SandboxFusion endpoint using ``requests``. All network calls
are wrapped with configurable retry logic (exponential back-off with jitter)
powered by the ``tenacity`` library.
"""

import logging
from typing import Optional
from functools import wraps
import asyncio

import requests
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from .common import trim_slash

from .models import RunCodeRequest, RunCodeResponse, EvalResult, \
    SubmitRequest, CommandRunStatus, RunStatus, SummaryMapping
from . import config

logger = logging.getLogger(__name__)


def set_endpoint(endpoint: str):
    """Override the global sandbox server endpoint at runtime.

    Args:
        endpoint: The full base URL of the SandboxFusion server
                  (e.g. ``"http://my-host:8080"``).
    """
    config.SANDBOX_ENDPOINT = endpoint


def on_retry_error(s):
    e = s.outcome.exception()
    logger.error(f'give up requesting sandbox. error: {e}')
    raise e


def before_retry_sleep(s):
    msg = f'error requesting sandbox for {s.attempt_number} time(s), will retry... error: {s.outcome.exception()}'
    if s.attempt_number > 2:
        logger.warning(msg)
    else:
        logger.debug(msg)


def configurable_retry(max_attempts):
    """Decorator factory that adds retry-with-exponential-jitter to a function.

    Wraps the target function with ``tenacity.retry``. Automatically detects
    whether the wrapped function is a coroutine and applies the appropriate
    sync or async wrapper.

    Args:
        max_attempts: Maximum number of attempts before giving up.

    Returns:
        A decorator that adds retry behaviour to the wrapped function.
    """

    def decorator(func):

        @wraps(func)
        @retry(wait=wait_exponential_jitter(),
               stop=stop_after_attempt(max_attempts),
               before_sleep=before_retry_sleep,
               retry_error_callback=on_retry_error)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        @wraps(func)
        @retry(wait=wait_exponential_jitter(),
               stop=stop_after_attempt(max_attempts),
               before_sleep=before_retry_sleep,
               retry_error_callback=on_retry_error)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def run_code(request: RunCodeRequest,
             endpoint: str = '',
             max_attempts: int = 5,
             client_timeout: Optional[float] = None) -> RunCodeResponse:
    """Execute code on the sandbox server.

    Sends a POST request to the ``/run_code`` endpoint. Retries automatically
    on transient errors and raises if the sandbox itself reports an error.

    Args:
        request: The code execution request payload.
        endpoint: Optional override for the server URL. Falls back to the
                  global ``SANDBOX_ENDPOINT`` when empty.
        max_attempts: Maximum number of retry attempts (default 5).
        client_timeout: Optional HTTP timeout in seconds.

    Returns:
        A :class:`RunCodeResponse` with compilation/run results.

    Raises:
        Exception: On non-200 HTTP status or a ``SandboxError`` response.
    """

    @configurable_retry(max_attempts)
    def _run_code(request: RunCodeRequest) -> RunCodeResponse:
        result = requests.post(f'{trim_slash(endpoint or config.SANDBOX_ENDPOINT)}/run_code',
                               json=request.dict(),
                               timeout=client_timeout)
        if result.status_code != 200:
            raise Exception(f'API responded with code {result.status_code}: {result.text}')
        resp = RunCodeResponse(**result.json())
        if resp.status == RunStatus.SandboxError:
            raise Exception(f'Sandbox responded with error: {resp.message}')
        return resp

    return _run_code(request)


def summary_run_code_result(result: RunCodeResponse, mapping: SummaryMapping) -> str:
    """Classify a code-execution response into a single summary status string.

    Inspects the compile and run results within *result* and maps the outcome
    to one of the status strings defined in *mapping* (e.g. Success, Failed,
    CompileTimeout, RunFailed, etc.).

    Args:
        result: The response from a ``run_code`` call.
        mapping: A :class:`SummaryMapping` that defines the string to return
                 for each possible outcome category.

    Returns:
        The summary status string corresponding to the outcome.

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


def submit(request: SubmitRequest,
           endpoint: str = '',
           max_attempts: int = 5,
           client_timeout: Optional[float] = None) -> EvalResult:
    """Submit code for evaluation against test cases.

    Sends a POST request to the ``/submit`` endpoint. The server extracts
    code from the completion, runs it against the provided test cases, and
    returns an :class:`EvalResult`.

    Args:
        request: The submission payload including completion and test cases.
        endpoint: Optional override for the server URL.
        max_attempts: Maximum number of retry attempts (default 5).
        client_timeout: Optional HTTP timeout in seconds.

    Returns:
        An :class:`EvalResult` indicating whether the submission was accepted.

    Raises:
        Exception: On non-200 HTTP status.
    """

    @configurable_retry(max_attempts)
    def _submit(request: SubmitRequest) -> EvalResult:
        result = requests.post(f'{trim_slash(endpoint or config.SANDBOX_ENDPOINT)}/submit',
                               json=request.dict(),
                               timeout=client_timeout)
        if result.status_code != 200:
            raise Exception(f'API responded with code {result.status_code}: {result.text}')
        resp = EvalResult(**result.json())
        return resp

    return _submit(request)


def submit_safe(request: SubmitRequest,
                endpoint: str = '',
                max_attempts: int = 5,
                client_timeout: Optional[float] = None) -> EvalResult:
    """Submit code for evaluation, returning a rejected result on failure.

    Identical to :func:`submit` but catches all exceptions and returns a
    synthetic rejected :class:`EvalResult` instead of propagating them.
    Useful in batch pipelines where one failing request should not abort
    the entire run.

    Args:
        request: The submission payload including completion and test cases.
        endpoint: Optional override for the server URL.
        max_attempts: Maximum number of retry attempts (default 5).
        client_timeout: Optional HTTP timeout in seconds.

    Returns:
        An :class:`EvalResult`. On error, ``accepted`` is ``False`` and
        ``tests`` is empty.
    """
    try:
        return submit(request, endpoint, max_attempts, client_timeout)
    except Exception:
        logger.warning('failed to request sandbox, a rejected result is returned')
        return EvalResult(id=request.id, accepted=False, extracted_code='', tests=[])
