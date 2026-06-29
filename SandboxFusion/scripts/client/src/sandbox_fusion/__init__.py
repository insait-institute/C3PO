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

"""SandboxFusion Python SDK.

Provides both synchronous and asynchronous clients for interacting with a
SandboxFusion server, along with all request/response model classes and
concurrency helpers.

Synchronous API:
    run_code, submit, submit_safe, summary_run_code_result, set_endpoint

Asynchronous API:
    run_code_async, submit_async, submit_safe_async

Models:
    RunCodeRequest, RunCodeResponse, EvalResult, SubmitRequest,
    GeneralStdioTest, CommandRunStatus, RunStatus, SummaryMapping, TestConfig

Concurrency helpers:
    run_concurrent, run_concurrent_pure
"""

from .client import run_code, summary_run_code_result, submit, \
    submit_safe, set_endpoint
from .async_client import run_code as run_code_async, submit as submit_async, \
    submit_safe as submit_safe_async
from .models import RunCodeRequest, RunCodeResponse, EvalResult, \
    SubmitRequest, GeneralStdioTest, \
    CommandRunStatus, RunStatus, \
    SummaryMapping, TestConfig
from .common import run_concurrent, run_concurrent_pure

__all__ = [
    'run_code',
    'summary_run_code_result',
    'submit',
    'submit_safe',
    'set_endpoint',
    'RunCodeRequest',
    'RunCodeResponse',
    'EvalResult',
    'SubmitRequest',
    'GeneralStdioTest',
    'CommandRunStatus',
    'RunStatus',
    'SummaryMapping',
    'TestConfig',
    'run_concurrent',
    'run_concurrent_pure',
    'run_code_async',
    'submit_async',
    'submit_safe_async',
]
