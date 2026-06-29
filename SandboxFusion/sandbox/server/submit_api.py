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

"""Evaluation submission API.

Defines the ``POST /submit`` endpoint that accepts a code completion,
extracts executable code from it using configurable extraction logic,
runs the code against a set of stdin/stdout test cases in parallel, and
returns an :class:`~sandbox.datasets.types.EvalResult` indicating
whether all tests passed.
"""

from fastapi import APIRouter, HTTPException

from sandbox.datasets.types import EvalResult, SubmitRequest
from sandbox.utils.extraction import default_extract_helper
from sandbox.utils.testing import check_stdio_test_cases_parallel

submit_router = APIRouter()


@submit_router.post("/submit", description='Submit code for evaluation against stdin/stdout test cases', tags=['eval'])
async def submit(request: SubmitRequest) -> EvalResult:
    """Evaluate a code completion against stdin/stdout test cases.

    Processing steps:

    1. Validate that ``request.config.language`` is set (raises 400 if
       missing).
    2. Extract executable code from the raw completion string using
       :func:`~sandbox.utils.extraction.default_extract_helper`, which
       applies language-specific heuristics and any custom extraction
       logic specified in the config.
    3. Run the extracted code against every test case in parallel via
       :func:`~sandbox.utils.testing.check_stdio_test_cases_parallel`.
    4. Return an :class:`~sandbox.datasets.types.EvalResult` whose
       ``accepted`` flag is ``True`` only when *all* test cases pass.

    Args:
        request: Validated submission payload containing the completion
            text, test cases, and execution configuration.

    Returns:
        An :class:`~sandbox.datasets.types.EvalResult` with the
        extracted code, per-test outcomes, and overall acceptance.

    Raises:
        HTTPException: ``400`` if ``config.language`` is not provided.
    """
    if not request.config.language:
        raise HTTPException(status_code=400, detail='config.language is required')

    code = default_extract_helper(request.completion, request.config.language, request.config.custom_extract_logic)
    outcomes = await check_stdio_test_cases_parallel(code, request.test_cases, request.config)
    return EvalResult(
        id=request.id,
        accepted=all(o.passed for o in outcomes),
        extracted_code=code,
        tests=outcomes,
    )
