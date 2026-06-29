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

"""Test case execution engine for evaluating code against expected outputs.

Provides functions to run code in the sandbox and verify results against test
cases. Supports both auto-test (pass/fail based on return code) and stdio-based
testing (comparing stdout against expected output with case-insensitive,
float-tolerant, whitespace-tolerant comparison). Includes sequential and
parallel test case runners, as well as a Jest report parser.
"""

import asyncio
import json
from typing import Any, Dict, List

import structlog
from fastapi import HTTPException

from sandbox.configs.run_config import RunConfig
from sandbox.datasets.types import EvalTestCase, GeneralStdioTest, RunStatus, TestConfig
from sandbox.runners.types import compile_languages
from sandbox.utils.common import truncate_str
from sandbox.utils.sandbox_client import RunCodeRequest, run_code_in_sandbox, run_code_in_sandbox_w_retry

eval_config = RunConfig.get_instance_sync()
logger = structlog.stdlib.get_logger()

_runner_semaphore: asyncio.Semaphore | None = None


def _get_runner_semaphore() -> asyncio.Semaphore | None:
    global _runner_semaphore
    if _runner_semaphore is None and eval_config.eval.max_runner_concurrency > 0:
        _runner_semaphore = asyncio.Semaphore(eval_config.eval.max_runner_concurrency)
    return _runner_semaphore


async def check_auto_test_case(code: str, config: TestConfig) -> EvalTestCase:
    """Run code in the sandbox and check if the return code is 0 (auto-test mode).

    Executes the given code without stdin and determines pass/fail based solely
    on whether the sandbox reports a ``Success`` status (i.e., zero exit code).

    Args:
        code: The source code to execute.
        config: Test configuration specifying the language and other settings.

    Returns:
        An ``EvalTestCase`` with ``passed=True`` if the execution succeeded,
        and the full execution info attached.
    """
    result = await run_code_in_sandbox(RunCodeRequest(code=code, language=config.language))
    return EvalTestCase(passed=result.status == RunStatus.Success, exec_info=result)


def is_float(s):
    """Check whether a string can be parsed as a float.

    Args:
        s: The string to test.

    Returns:
        ``True`` if ``float(s)`` succeeds, ``False`` otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def float_equal(a, b, rel_tol=1e-5):
    """Check whether two floats are approximately equal using relative tolerance.

    Computes the relative difference as ``|a - b| / max(|b|, 1e-10)`` and
    compares against the tolerance.

    Args:
        a: First float value.
        b: Second float value (used as the reference for relative comparison).
        rel_tol: Relative tolerance threshold. Defaults to ``1e-5``.

    Returns:
        ``True`` if the values are within the relative tolerance, ``False``
        otherwise.
    """
    return abs(a - b) / max(abs(b), 1e-10) < rel_tol


async def check_stdio_test_case(code: str, case: GeneralStdioTest, config: TestConfig, lower_cmp=True) -> EvalTestCase:
    """Run code with stdin input and compare stdout against expected output.

    Executes the code in the sandbox with the test case's stdin, then compares
    the actual stdout to the expected stdout line-by-line. Comparison features:

    - Case-insensitive matching (when ``lower_cmp=True``).
    - Whitespace-tolerant (strips each line before comparing).
    - Float-tolerant (if both actual and expected lines parse as floats,
      uses approximate comparison via :func:`float_equal`).
    - Handles trailing blank lines gracefully.

    For compiled languages, uses retry-enabled sandbox execution with separate
    compile and run timeouts.

    Args:
        code: The source code to execute.
        case: The test case containing ``input['stdin']`` and
            ``output['stdout']``.
        config: Test configuration specifying language, timeouts, and extra
            options. If ``extra['return_full_case']`` is False, input/output
            strings are truncated in the returned test info.
        lower_cmp: Whether to lowercase both actual and expected output before
            comparing. Defaults to ``True``.

    Returns:
        An ``EvalTestCase`` with ``passed`` indicating whether all output lines
        matched, along with execution info and test case details.
    """
    if config.language in compile_languages:
        result = await run_code_in_sandbox_w_retry(
            RunCodeRequest(code=code,
                           language=config.language,
                           stdin=case.input['stdin'],
                           compile_timeout=config.compile_timeout or 10,
                           run_timeout=config.run_timeout or 10))
    else:
        result = await run_code_in_sandbox_w_retry(
            RunCodeRequest(code=code,
                           language=config.language,
                           stdin=case.input['stdin'],
                           run_timeout=config.run_timeout or 20))
    fail_case = EvalTestCase(passed=False, exec_info=result, test_info=case.model_dump())
    if result.status != 'Success':
        return fail_case
    result_lines = result.run_result.stdout.strip().split('\n')
    expected_lines = case.output['stdout'].strip().split('\n')
    if len(result_lines) - len(expected_lines) == 1 and result_lines[-1] == '':
        result_lines = result_lines[:-1]
    if len(expected_lines) - len(result_lines) == 1 and expected_lines[-1] == '':
        expected_lines = expected_lines[:-1]
    if len(result_lines) != len(expected_lines):
        return fail_case
    for rl, el in zip(result_lines, expected_lines):
        if lower_cmp:
            rl = rl.lower()
            el = el.lower()
        if rl.strip() != el.strip():
            if is_float(el) and is_float(rl):
                if float_equal(float(rl), float(el)):
                    continue
            return fail_case
    if not config.extra.get('return_full_case', False):
        for k in case.input:
            case.input[k] = truncate_str(case.input[k])
        for k in case.output:
            case.output[k] = truncate_str(case.output[k])
    return EvalTestCase(passed=True, exec_info=result, test_info=case.model_dump())


async def check_stdio_test_cases(code: str,
                                 cases: List[GeneralStdioTest],
                                 config: TestConfig,
                                 lower_cmp=True) -> List[EvalTestCase]:
    """Run test cases sequentially, stopping on the first failure.

    Executes each test case in order via :func:`check_stdio_test_case`. If any
    test case fails, execution stops immediately and the results so far are
    returned.

    Args:
        code: The source code to execute for each test case.
        cases: The list of stdio test cases to run.
        config: Test configuration specifying language, timeouts, and options.
        lower_cmp: Whether to lowercase output before comparing. Defaults
            to ``True``.

    Returns:
        A list of ``EvalTestCase`` results, ending at the first failure
        (or containing all results if all passed).
    """
    result = []
    for case in cases:
        outcome = await check_stdio_test_case(code, case, config, lower_cmp)
        result.append(outcome)
        if not outcome.passed:
            break
    return result


async def check_stdio_test_cases_parallel(code: str,
                                          cases: List[GeneralStdioTest],
                                          config: TestConfig,
                                          lower_cmp=True) -> List[EvalTestCase]:
    """Run test cases in parallel using asyncio tasks.

    Creates an asyncio task for each test case and awaits them in order. If
    ``max_runner_concurrency`` is configured in ``RunConfig.eval``, applies a
    concurrency limit via a shared semaphore. By default, stops on the
    first failure and cancels remaining tasks; set ``config.extra['run_all_cases']``
    to ``True`` to continue past failures.

    Args:
        code: The source code to execute for each test case.
        cases: The list of stdio test cases to run in parallel.
        config: Test configuration specifying language, timeouts, and options.
            The ``extra['run_all_cases']`` flag controls whether to continue
            after a failure.
        lower_cmp: Whether to lowercase output before comparing. Defaults
            to ``True``.

    Returns:
        A list of ``EvalTestCase`` results.

    Raises:
        HTTPException: With status 500 if a task raises an unexpected exception.
    """
    result = []
    tasks: List[asyncio.Task[EvalTestCase]] = []

    sem = _get_runner_semaphore()

    async def _run_limited(code, case, config, lower_cmp):
        if sem is not None:
            async with sem:
                return await check_stdio_test_case(code, case, config, lower_cmp)
        return await check_stdio_test_case(code, case, config, lower_cmp)

    for case in cases:
        task = asyncio.create_task(_run_limited(code, case, config, lower_cmp))
        tasks.append(task)

    run_all_cases = config.extra.get("run_all_cases", False)

    for task in tasks:
        try:
            outcome = await task
        except Exception as e:
            raise HTTPException(status_code=500, detail=f'Failed to check stdio test case: {e}')
        result.append(outcome)

        if not run_all_cases and not outcome.passed:
            for remaining_task in tasks:
                if not remaining_task.done():
                    remaining_task.cancel()
            break

    return result


def parse_jest_cases(report_data: str) -> List[Dict[str, Any]]:
    """Parse a Jest JSON report into a list of test case dictionaries.

    Extracts individual test case results from a Jest JSON report, including
    pass/fail status, full test name, file path, test suite hierarchy, and
    any failure messages.

    Args:
        report_data: Either a JSON string or an already-parsed dict/list
            containing the Jest report with ``testResults`` entries.

    Returns:
        A list of dictionaries, each containing:
            - ``passed`` (bool): Whether the test case passed.
            - ``full_name`` (str): The full test name.
            - ``file`` (str): The test file path.
            - ``suite`` (str): The ancestor test suite titles joined by `` > ``.
            - ``test`` (str): The individual test title.
            - ``failure_messages`` (list): Any failure message strings.
    """
    if isinstance(report_data, str):
        report = json.loads(report_data)
    else:
        report = report_data

    test_cases = []

    for test_suite in report['testResults']:
        file_path = test_suite['testFilePath']

        for test_case in test_suite['testResults']:
            result = {
                'passed': test_case['status'] == 'passed',
                'full_name': test_case['fullName'],
                'file': file_path,
                'suite': ' > '.join(test_case['ancestorTitles']),
                'test': test_case['title'],
                'failure_messages': test_case['failureMessages']
            }
            test_cases.append(result)

    return test_cases
