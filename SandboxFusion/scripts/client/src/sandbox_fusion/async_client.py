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

"""Asynchronous HTTP client for the SandboxFusion server.

Provides ``async`` versions of the core client operations (:func:`run_code`,
:func:`submit`, :func:`submit_safe`) using ``aiohttp``. Retry behaviour is
inherited from :func:`sandbox_fusion.client.configurable_retry`.
"""

import aiohttp
import logging
from typing import Optional

from .common import trim_slash
from .client import configurable_retry
from . import config

from .models import RunCodeRequest, RunCodeResponse, EvalResult, \
    SubmitRequest, RunStatus

logger = logging.getLogger(__name__)


async def run_code(request: RunCodeRequest,
                   endpoint: str = '',
                   max_attempts: int = 5,
                   client_timeout: Optional[float] = None) -> RunCodeResponse:
    """Execute code on the sandbox server (async).

    Async equivalent of :func:`sandbox_fusion.client.run_code`. Sends a POST
    to ``/run_code`` via ``aiohttp`` with automatic retry on transient errors.

    Args:
        request: The code execution request payload.
        endpoint: Optional override for the server URL.
        max_attempts: Maximum number of retry attempts (default 5).
        client_timeout: Optional HTTP timeout in seconds.

    Returns:
        A :class:`RunCodeResponse` with compilation/run results.

    Raises:
        Exception: On non-200 HTTP status or a ``SandboxError`` response.
    """

    @configurable_retry(max_attempts)
    async def _run_code(request: RunCodeRequest) -> RunCodeResponse:
        timeout = aiohttp.ClientTimeout(total=client_timeout) if client_timeout else None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f'{trim_slash(endpoint or config.SANDBOX_ENDPOINT)}/run_code',
                                    json=request.dict()) as result:
                if result.status != 200:
                    raise Exception(f'API responded with code {result.status}: {await result.text()}')
                resp = RunCodeResponse(**(await result.json()))
                if resp.status == RunStatus.SandboxError:
                    raise Exception(f'Sandbox responded with error: {resp.message}')
                return resp

    return await _run_code(request)


async def submit(request: SubmitRequest,
                 endpoint: str = '',
                 max_attempts: int = 5,
                 client_timeout: Optional[float] = None) -> EvalResult:
    """Submit code for evaluation against test cases (async).

    Async equivalent of :func:`sandbox_fusion.client.submit`. Sends a POST
    to ``/submit`` via ``aiohttp``.

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
    async def _submit(request: SubmitRequest) -> EvalResult:
        timeout = aiohttp.ClientTimeout(total=client_timeout) if client_timeout else None
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f'{trim_slash(endpoint or config.SANDBOX_ENDPOINT)}/submit',
                                    json=request.dict()) as result:
                if result.status != 200:
                    raise Exception(f'API responded with code {result.status}: {await result.text()}')
                resp = EvalResult(**(await result.json()))
                return resp

    return await _submit(request)


async def submit_safe(request: SubmitRequest,
                      endpoint: str = '',
                      max_attempts: int = 5,
                      client_timeout: Optional[float] = None) -> EvalResult:
    """Submit code for evaluation, returning a rejected result on failure (async).

    Async equivalent of :func:`sandbox_fusion.client.submit_safe`. Catches
    all exceptions and returns a synthetic rejected :class:`EvalResult`.

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
        return await submit(request, endpoint, max_attempts, client_timeout)
    except Exception:
        logger.warning('failed to request sandbox, a rejected result is returned')
        return EvalResult(id=request.id, accepted=False, extracted_code='', tests=[])
