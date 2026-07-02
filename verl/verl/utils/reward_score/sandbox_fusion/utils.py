# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import concurrent.futures  # <-- Import concurrent.futures
import json
import logging
import math
import os
import random
import threading
import time
import traceback
import uuid
from typing import Any, Optional

import requests

DEFAULT_TIMEOUT = 10  # Default compile and run timeout
MAX_RETRIES = 3  # attempts when retry_on_timeout is False (legacy fast-fail path)
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10

# Retry tuning for the retry_on_timeout=True path.  There are two distinct
# transient failure modes and they want different backoffs:
#
#   * overload   -- HTTP 503 (the server's admission control rejected us
#     because every execution slot is busy) or 504 (gateway timeout).  The
#     server is healthy, just saturated; its queue drains in seconds, so retry
#     fast.
#   * unreachable -- a read timeout or connection error.  With server-side
#     admission control these are now rare and usually mean the server is
#     actually down/restarting (the standing-server supervisor restart + boot
#     takes a couple of minutes), so back off more.
#
# Both use capped exponential backoff with *full jitter*.  Jitter matters: a
# single reward step times out many cases at once, and a fixed delay makes them
# all retry on the same tick and re-saturate the server; jitter spreads the
# retries out.  The total budget is kept well under the trainer's 600s NCCL
# collective-timeout watchdog so one stuck case can never stall a reward step
# long enough to crash training.  All knobs are env-overridable.
MAX_TIMEOUT_RETRIES = int(os.getenv("SANDBOX_FUSION_MAX_RETRIES", "6"))
OVERLOAD_BACKOFF_BASE = float(os.getenv("SANDBOX_FUSION_OVERLOAD_BACKOFF_BASE", "1.0"))
OVERLOAD_BACKOFF_CAP = float(os.getenv("SANDBOX_FUSION_OVERLOAD_BACKOFF_CAP", "8.0"))
UNREACHABLE_BACKOFF_BASE = float(os.getenv("SANDBOX_FUSION_UNREACHABLE_BACKOFF_BASE", "2.0"))
UNREACHABLE_BACKOFF_CAP = float(os.getenv("SANDBOX_FUSION_UNREACHABLE_BACKOFF_CAP", "30.0"))

# A sandbox run-timeout (run_result.status == "TimeLimitExceeded") comes back as a
# *successful* HTTP 200 response, so the call_sandbox_api retry path above (which
# only covers transport-level read timeouts / 503 / 504) never sees it -- the
# verdict is otherwise baked into the case's score as a final failure.  Under a
# CPU-throughput-bound sandbox these verdicts have a transient component (a case
# that needs ~the limit of CPU can tip over the wall-clock limit when the server
# is momentarily contended), which injects correctness-uncorrelated noise into
# the reward.  Re-run a case that returns TimeLimitExceeded a few times before
# accepting it; a genuinely-too-slow solution still times out every attempt and
# correctly fails, while a transient one gets a chance to pass.  Set to 0 to
# disable (legacy behavior).
TLE_RETRIES = int(os.getenv("SANDBOX_FUSION_TLE_RETRIES", "2"))


def _is_run_timeout(api_response: Optional[dict[str, Any]]) -> bool:
    """True iff the sandbox reported a run-phase TimeLimitExceeded verdict."""
    if not api_response:
        return False
    run_result = api_response.get("run_result") or {}
    return api_response.get("status") == "Failed" and run_result.get("status") == "TimeLimitExceeded"


def _backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Capped exponential backoff with full jitter: uniform(0, min(cap, base*2**attempt))."""
    return random.uniform(0, min(cap, base * (2**attempt)))

logger = logging.getLogger(__name__)

# Define supported languages list (optional, for documentation or validation)
SUPPORTED_LANGUAGES = [
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
]


def call_sandbox_api(
    sandbox_fusion_url: str,
    code: str,
    stdin: Optional[str],
    compile_timeout: int,
    run_timeout: int,
    memory_limit_mb: int,
    language: str = "python",
    retry_on_timeout: bool = True,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:  # <-- Remove request_id parameter
    """
    Calls the remote sandbox API to execute code with retry logic for Gateway Timeout,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        code: The code string to execute.
        stdin: The standard input string.
        compile_timeout: Compile timeout in seconds.
        run_timeout: Run timeout in seconds.
        language: The programming language of the code (e.g., "python", "cpp", "java"). Defaults to "python".
        retry_on_timeout: If True, overloaded responses (HTTP 503/504), read
            timeouts and connection errors are retried with jittered, capped
            exponential backoff (see MAX_TIMEOUT_RETRIES / *_BACKOFF_* knobs) so
            server saturation or a restart shows up as a slow reward instead of a
            failed case. If False, they fail the request immediately (legacy
            behavior).

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())  # <-- Generate request_id internally
    log_prefix = f"[Request ID: {request_id}] "  # <-- Create log prefix

    if language not in SUPPORTED_LANGUAGES:
        error_msg = f"{log_prefix}Unsupported language: {language}"
        logger.error(error_msg)
        return None, error_msg

    payload = json.dumps(
        {
            "compile_timeout": compile_timeout,
            "run_timeout": run_timeout,
            "code": code,
            "stdin": stdin,
            "memory_limit_MB": memory_limit_mb,
            "language": language,  # Use the passed language parameter
            "files": {},
            "fetch_files": [],
        }
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    # Calculate a reasonable request timeout based on compile/run timeouts plus a buffer
    request_timeout = compile_timeout + run_timeout + API_TIMEOUT

    last_error = None  # Store the last error encountered
    max_attempts = MAX_TIMEOUT_RETRIES if retry_on_timeout else MAX_RETRIES

    for attempt in range(max_attempts):
        try:
            logger.info(f"{log_prefix}Attempt {attempt + 1}/{max_attempts}: Calling sandbox API at {sandbox_fusion_url}")  # <-- Use internal log_prefix
            response = requests.post(
                sandbox_fusion_url,
                headers=headers,
                data=payload,
                timeout=request_timeout,  # Use the calculated timeout
            )

            # Overload responses: 503 (admission control rejected -- all slots
            # busy) and 504 (gateway timeout).  Server is healthy but saturated;
            # retry fast.
            if response.status_code in (503, 504):
                last_error = (
                    f"{log_prefix}API Request Error: HTTP {response.status_code} "
                    f"(server overloaded) on attempt {attempt + 1}/{max_attempts}"
                )
                logger.warning(last_error)
                if attempt < max_attempts - 1:  # Don't sleep after the last attempt
                    if retry_on_timeout:
                        delay = _backoff_delay(attempt, OVERLOAD_BACKOFF_BASE, OVERLOAD_BACKOFF_CAP)
                    else:
                        delay = INITIAL_RETRY_DELAY * (attempt + 1)  # legacy linear path
                    logger.info(f"{log_prefix}Retrying after {delay:.1f} seconds...")  # <-- Use internal log_prefix
                    time.sleep(delay)
                continue  # Go to the next retry attempt

            # Check for other HTTP errors (e.g., 4xx, other 5xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Sandbox API call successful on attempt {attempt + 1}")  # <-- Use internal log_prefix
            return response.json(), None

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            # Read timeouts / connection errors now usually mean the server is
            # actually down or restarting (admission control turns mere
            # saturation into fast 503s above).  Back off with jittered capped
            # exponential delay -- enough to ride a brief restart, but bounded
            # so a stuck case can't stall the reward step into the NCCL watchdog.
            last_error = f"{log_prefix}API Request Error: {e}"  # <-- Use internal log_prefix
            if not retry_on_timeout:
                break
            logger.warning(f"{log_prefix}Timeout/connection error on attempt {attempt + 1}/{max_attempts}: {e}")
            if attempt < max_attempts - 1:  # Don't sleep after the last attempt
                delay = _backoff_delay(attempt, UNREACHABLE_BACKOFF_BASE, UNREACHABLE_BACKOFF_CAP)
                logger.info(f"{log_prefix}Retrying after {delay:.1f} seconds...")  # <-- Use internal log_prefix
                time.sleep(delay)
            continue  # Go to the next retry attempt
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on non-504 request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"  # <-- Use internal log_prefix
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Sandbox API call failed. Last error: {last_error}")  # <-- Use internal log_prefix
    # Return the error message without the prefix, as the caller doesn't need the internal ID
    # Ensure API call failure returns error message, leading to -1 in check_correctness
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _normalize_output(s: Any) -> str:
    """Canonicalise program output the way competitive-programming judges do.

    Unifies line endings, strips trailing whitespace on every line, and drops
    trailing blank lines.  Interior layout (line breaks, intra-line spacing) is
    preserved so genuinely wrong-shaped answers still fail.
    """
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in s.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _outputs_match(actual: Any, expected: Any) -> bool:
    """Judge a stdin/stdout case at parity with the in-tree ``prime_code`` checker.

    The remote-sandbox path historically compared with a bare ``rstrip("\\n")``
    exact match, which marks correct solutions wrong on trivial formatting
    differences (per-line trailing whitespace) and makes floating-point answers
    essentially unjudgeable ("0.5" vs "0.500000").  That injects
    correctness-uncorrelated noise into the reward and caps GRPO's learning
    signal.

    Matching logic, in order:
      1. exact match after per-line normalisation (handles whitespace/newlines);
      2. numeric fallback with ``math.isclose`` (rel_tol 1e-5, abs_tol 1e-6,
         mirroring prime_code's ``np.allclose``), applied token-by-token
         *within the same line layout* -- same line count and same token count
         per line -- so a float answer that differs only in precision passes
         while a differently-shaped answer still fails.
    """
    na, ne = _normalize_output(actual), _normalize_output(expected)
    if na == ne:
        return True

    lines_a, lines_e = na.split("\n"), ne.split("\n")
    if len(lines_a) != len(lines_e):
        return False
    try:
        for row_a, row_e in zip(lines_a, lines_e):
            toks_a, toks_e = row_a.split(), row_e.split()
            if len(toks_a) != len(toks_e):
                return False
            for a, b in zip(toks_a, toks_e):
                # Non-numeric tokens must match exactly; numeric ones within tol.
                if a != b and not math.isclose(float(a), float(b), rel_tol=1e-5, abs_tol=1e-6):
                    return False
        return True
    except ValueError:
        # A token that isn't a float and didn't match exactly -> genuine mismatch.
        return False


def _cancelled_result(case_index: int, expected_output: Any) -> tuple[None, dict[str, Any]]:
    """Sentinel returned for a case skipped by short-circuit cancellation.

    result_status is None -- not True -- so it never counts as a pass; it just
    marks that the case was abandoned once the binary verdict was already decided.
    """
    return None, {
        "case_index": case_index,
        "input": None,
        "expected_output": str(expected_output) if expected_output else None,
        "api_request_error": None,
        "status": "cancelled",
    }


def _process_single_case(
    case_index: int,
    stdin_data: Any,
    expected_output: Any,
    sandbox_fusion_url: str,
    generation: str,
    timeout: int,
    memory_limit_mb: int,
    language: str,
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    fn_name: Optional[str] = None,
    retry_on_timeout: bool = True,
    stop_event: Optional[threading.Event] = None,
) -> tuple[int, dict[str, Any]]:
    """Helper function to process a single test case."""
    api_response = None
    error_msg = None
    logger.info(f"Processing test case {case_index + 1}.")

    # Short-circuit: a sibling case already failed, so the binary verdict is
    # settled -- don't queue for a sandbox slot at all.
    if stop_event is not None and stop_event.is_set():
        return _cancelled_result(case_index, expected_output)

    current_generation_code = generation

    if fn_name and language == "python":
        # Wrapper assumes stdin_data is a JSON string for function arguments.
        wrapper_code = f"""
import traceback
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json

# === User's Original Code START ===
{generation}
# === User's Original Code END ===

_SANDBOX_FN_NAME = "{fn_name}"

def _execute_user_function():
    # --- Input Parsing ---
    _raw_input_str = sys.stdin.read()
    _args = []
    if _raw_input_str.strip(): # If there's input
        try:
            _args = [json.loads(line) for line in _raw_input_str.split('\\n')]
        except json.JSONDecodeError as _je:
            sys.stderr.write(f"WrapperError: Invalid JSON input for '{{_SANDBOX_FN_NAME}}': {{_je}}\\nInput was: "
                              f"{{_raw_input_str[:200]}}\\n")
            return None, True # result, error_occurred

    # --- Function Location and Execution ---
    try:
        _target_callable = None
        # Try global scope first
        if _SANDBOX_FN_NAME in globals():
            _target_callable = globals()[_SANDBOX_FN_NAME]
        # Else, if 'Solution' class exists, try to get its method
        elif 'Solution' in globals():
            _Solution_class = globals()['Solution']
            # Attempt to instantiate and get method.
            # Errors (e.g., Solution not a class, instantiation fails, method missing)
            # will be caught by the broad except block below.
            _solution_instance = _Solution_class()
            _target_callable = getattr(_solution_instance, _SANDBOX_FN_NAME)

        if not _target_callable:
            sys.stderr.write(f"WrapperError: Function or method '{{_SANDBOX_FN_NAME}}' not found.\\n")
            return None, True # result, error_occurred

        _fn_result = _target_callable(*_args)
        return _fn_result, False # result, no_error
    except Exception: # Catches errors from Solution instantiation, getattr, or function call
        sys.stderr.write(f"Error during setup or execution of '{{_SANDBOX_FN_NAME}}':\\n{{traceback.format_exc()}}\\n")
        return None, True # result, error_occurred

if __name__ == '__main__':
    _result, _error_occurred = _execute_user_function()

    if not _error_occurred:
        # Serialize result to stdout
        if isinstance(_result, (dict, list, tuple)) or _result is None or isinstance(_result, bool):
            print(json.dumps(_result))
        elif isinstance(_result, (int, float, str)):
            print(str(_result)) # Ensure string conversion for print
        else:
            # For other types, default to string representation.
            print(str(_result))
    # Optional: To explicitly exit with an error code if the sandbox relies on it
    # else:
    #    sys.exit(1)
"""
        current_generation_code = wrapper_code

    stdin = None if stdin_data is None else str(stdin_data)

    def _invoke_sandbox() -> tuple[Optional[dict[str, Any]], Optional[str]]:
        return call_sandbox_api(
            sandbox_fusion_url=sandbox_fusion_url,
            code=current_generation_code,
            stdin=stdin,
            compile_timeout=timeout,
            run_timeout=timeout,
            memory_limit_mb=memory_limit_mb,
            language=language,
            retry_on_timeout=retry_on_timeout,
        )

    try:
        # Retry a run-timeout verdict up to TLE_RETRIES times (see _is_run_timeout):
        # the sandbox returns it as a successful HTTP response, so call_sandbox_api's
        # own retry never covers it.  The semaphore is re-acquired each attempt so a
        # retry waits its turn instead of holding a slot across the whole loop.
        for tle_attempt in range(TLE_RETRIES + 1):
            if concurrent_semaphore:
                with concurrent_semaphore:
                    # Re-check after possibly blocking on the semaphore: a failure may
                    # have landed while we waited for a slot.  Don't spend it.
                    if stop_event is not None and stop_event.is_set():
                        return _cancelled_result(case_index, expected_output)
                    api_response, error_msg = _invoke_sandbox()
            else:
                api_response, error_msg = _invoke_sandbox()

            if error_msg is None and tle_attempt < TLE_RETRIES and _is_run_timeout(api_response):
                # Don't bother re-running if the verdict is already settled elsewhere.
                if stop_event is not None and stop_event.is_set():
                    break
                logger.info(f"Case {case_index + 1}: run-timeout verdict, retry {tle_attempt + 1}/{TLE_RETRIES}.")
                continue
            break
    except Exception as e:
        error_msg = f"API Request Exception during check_correctness for case {case_index + 1}: {e}"
        logger.error(f"Case {case_index + 1}: {error_msg}")
        traceback.print_exc()

    metadata = {
        "case_index": case_index,
        "input": stdin,
        "expected_output": str(expected_output) if expected_output else None,
        "api_request_error": error_msg,
        "api_response": None,
        "status": "unknown",
        "stdout": None,
        "stderr": None,
        "exit_code": None,
        "duration": None,
        "compile_duration": None,
        "compile_stderr": None,
        "api_status": None,
        "compile_status": None,
        "run_status": None,
    }
    result_status = -1  # Default error: API request error or unknown sandbox error

    if error_msg:
        metadata["status"] = "api_error"
        result_status = -1  # API request itself failed (includes timeout after retries)
        logger.error(f"Case {case_index}: API error occurred: {error_msg}")
        # Log code and input only on error for brevity
        generation_to_log = generation[:200] + "..." if len(generation) > 200 else generation
        stdin_to_log = str(stdin)[:10] + "..." if len(str(stdin)) > 10 else str(stdin)
        logger.debug(f"Case {case_index}: code: {generation_to_log}")
        logger.debug(f"Case {case_index}: input: {stdin_to_log}")
    elif api_response:
        # --- Add debug logging ---
        logger.debug(f"Case {case_index}: API Response: {api_response}")
        metadata["api_response"] = api_response
        metadata["api_status"] = api_response.get("status")
        compile_result = api_response.get("compile_result")
        run_result = api_response.get("run_result")

        # Extract compile information
        if compile_result:
            metadata["compile_status"] = compile_result.get("status")
            metadata["compile_duration"] = compile_result.get("execution_time")
            metadata["compile_stderr"] = compile_result.get("stderr")

        # Extract run information
        if run_result:
            metadata["run_status"] = run_result.get("status")
            metadata["stdout"] = run_result.get("stdout")
            metadata["stderr"] = run_result.get("stderr")  # stderr during runtime
            metadata["exit_code"] = run_result.get("return_code")
            metadata["duration"] = run_result.get("execution_time")

        # --- Determine status based on API response ---
        api_status = metadata["api_status"]

        if api_status == "SandboxError":
            metadata["status"] = "sandbox_error"
            result_status = -1  # Internal sandbox error
        elif api_status == "Failed":
            # --- Add debug logging ---
            logger.debug(f"API returned Failed status. Response: {api_response}")
            logger.debug(f"Compile Result: {compile_result}")
            logger.debug(f"Run Result: {run_result}")
            # --- Check the logic here ---
            # Compile failed or timed out
            is_compile_error = compile_result and (
                metadata["compile_status"] in ["Error", "TimeLimitExceeded"] or (metadata["compile_status"] == "Finished" and compile_result.get("return_code") != 0)
            )
            if is_compile_error:
                # Differentiate between compile_error and compile_timeout based on specific status
                if metadata["compile_status"] == "TimeLimitExceeded":
                    metadata["status"] = "compile_timeout"
                else:  # Includes Error and Finished but return_code != 0 cases
                    metadata["status"] = "compile_error"
                result_status = -4
            # Run failed or timed out
            elif run_result:
                # Modified condition: Check for TimeLimitExceeded OR (Finished with non-zero exit code) OR Error status
                is_runtime_error = (
                    metadata["run_status"] == "TimeLimitExceeded"
                    or metadata["run_status"] == "Error"
                    or (metadata["run_status"] == "Finished" and run_result.get("return_code") != 0)
                )
                if is_runtime_error:
                    if metadata["run_status"] == "TimeLimitExceeded":
                        metadata["status"] = "timeout"  # Runtime timeout
                        result_status = -3
                    else:  # Includes Error and Finished with non-zero return_code
                        metadata["status"] = "runtime_error"
                        result_status = -2
                else:
                    # Other Failed status with run_result, classify as unknown failure
                    logger.warning(f"Unknown run_status '{metadata['run_status']}' or state within Failed API status.")
                    metadata["status"] = "unknown_failure"
                    result_status = -1  # Default to -1
            else:
                # Status is Failed but neither a clear compile error nor run_result exists
                logger.warning("API status Failed but cannot determine specific error type (compile/run).")
                metadata["status"] = "unknown_failure_state"
                result_status = -1  # Default to -1
        elif api_status == "Success":
            # Run completed successfully, now check the answer
            if run_result and metadata["run_status"] == "Finished":
                actual_output = metadata["stdout"] if metadata["stdout"] is not None else ""
                # Judge with competitive-programming normalisation (per-line
                # strip + float tolerance), at parity with the in-tree
                # prime_code checker -- a bare rstrip("\n") exact match here
                # marks correct solutions wrong on formatting/float differences
                # and flattens the GRPO reward signal.
                if expected_output is None or _outputs_match(actual_output, expected_output):
                    result_status = True
                    metadata["status"] = "success"
                else:
                    result_status = False
                    metadata["status"] = "wrong_answer"
            else:
                # Status is Success but run_result status is not Finished, this is unexpected
                metadata["status"] = "unexpected_success_state"
                result_status = -1  # Classify as unknown error
        else:
            # API returned an unknown top-level status
            logger.warning(f"Unknown API status received: {api_status}")
            metadata["status"] = f"unknown_api_status_{api_status}"
            result_status = -1  # Default to -1
    else:  # api_response is None and no error_msg (Should not happen with current call_sandbox_api logic)
        metadata["status"] = "unknown_api_state"
        result_status = -1
        logger.error(f"Case {case_index}: Unknown API state (no response and no error message).")
    return result_status, metadata


def check_correctness(
    sandbox_fusion_url: str,
    in_outs: Optional[dict],
    generation: str,
    timeout: int = DEFAULT_TIMEOUT,
    memory_limit_mb: int = 1024,
    language: str = "python",
    concurrent_semaphore: Optional[threading.Semaphore] = None,
    retry_on_timeout: bool = True,
    short_circuit: bool = False,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """
    Checks the correctness of code generation using the remote sandbox API,
    processing test cases concurrently.

    Args:
        sandbox_fusion_url: The URL of the sandbox fusion API.
        in_outs: Dictionary containing "inputs" and "outputs" lists.
        generation: The generated code string.
        timeout: Timeout for each test case (compile and run share this timeout).
        language: The programming language of the code.
        retry_on_timeout: Whether read timeouts / connection errors are retried
            with backoff instead of immediately failing the case (see
            call_sandbox_api).
        short_circuit: If True, stop as soon as one case is not a pass. The
            remaining cases are cancelled (queued ones never call the sandbox;
            unstarted futures are dropped). Used by the binary (all-must-pass)
            verdict, where a single failure already decides the score is 0.

    Returns:
        A tuple (results, metadata_list).
        results: A list containing the test result for each input/output pair
                 (True/False/-1 api/sandbox err, -2 runtime err, -3 timeout, -4 compile err).
                 Results are ordered corresponding to the inputs.
        metadata_list: A list containing metadata dictionaries for each test case,
                       ordered corresponding to the inputs.
    """
    logger.info("Starting correctness check for generation.")

    if not in_outs or "inputs" not in in_outs or "outputs" not in in_outs:
        logger.warning("Invalid in_outs format provided.")
        return [-1], [{"error": "Invalid input/output data"}]

    inputs = in_outs["inputs"]
    expected_outputs = in_outs["outputs"]
    fn_name = in_outs.get("fn_name")
    num_cases = len(inputs)
    assert_cases = in_outs.get("assert_case", [""] * num_cases)  # Default to empty strings if not provided
    results = [None] * num_cases  # Initialize with placeholders
    metadata_list = [None] * num_cases  # Initialize with placeholders

    if num_cases == 0:
        logger.warning("Empty inputs provided.")
        return [], []

    if len(inputs) != len(expected_outputs):
        logger.warning(f"Mismatch between number of inputs ({len(inputs)}) and outputs ({len(expected_outputs)}).")
        # Return error based on the number of inputs provided
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    # If assert_cases is provided, it overrides inputs and outputs
    if len(assert_cases) != num_cases:
        logger.warning(f"Mismatch between number of assert cases ({len(assert_cases)}) and inputs/outputs ({num_cases}).")
        return [-1] * num_cases, [{"error": "Input/output count mismatch", "case_index": i} for i in range(num_cases)]

    first_compile_error_index = -1
    # When short-circuiting, this is set on the first non-pass so still-queued
    # cases bail before spending a sandbox slot (checked in _process_single_case).
    stop_event = threading.Event() if short_circuit else None

    # Actual sandbox concurrency is gated by `concurrent_semaphore`
    # (reward.sandbox_fusion.max_concurrent), so any worker beyond that just
    # blocks on the semaphore while still costing a thread stack + its own glibc
    # malloc arena. The old `max(32, os.cpu_count() * 5)` spawned ~560 threads
    # per call on a many-core node; with ~32 concurrent check_correctness calls
    # in flight that ballooned to ~20k threads per RewardLoop worker, and the
    # freed per-arena memory was never returned to the OS (RSS ratcheted up
    # ~GB/step until OOM). Cap at the number of cases and a small ceiling; the
    # semaphore enforces the real sandbox concurrency limit.
    max_workers = max(1, min(num_cases, 32))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks, passing the concurrent_semaphore to _process_single_case
        future_to_index = {
            executor.submit(
                _process_single_case,
                i,
                stdin_data,
                expected_outputs[i],
                sandbox_fusion_url,
                generation + "\n\n" + assert_cases[i],  # Append assert case to generation
                timeout,
                memory_limit_mb,
                language,
                concurrent_semaphore,
                fn_name,
                retry_on_timeout,
                stop_event,
            ): i
            for i, stdin_data in enumerate(inputs)
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_status, metadata = future.result()
                results[index] = result_status
                metadata_list[index] = metadata

                # Check for compile error (-4)
                if result_status == -4:
                    if first_compile_error_index == -1 or index < first_compile_error_index:
                        first_compile_error_index = index
                    # Optimization: could potentially cancel futures for index > first_compile_error_index
                    # However, cancellation is not guaranteed. Post-processing is safer.

                # Binary short-circuit: the first non-pass settles the verdict
                # (0.0).  Signal queued cases to bail and stop waiting on the rest
                # so a wrong solution doesn't run every remaining test case.
                if short_circuit and result_status is not True:
                    stop_event.set()
                    break

            except Exception as exc:
                logger.error(f"Test case {index} generated an exception: {exc}")
                traceback.print_exc()
                results[index] = -1  # Mark as API/internal error
                metadata_list[index] = {
                    "case_index": index,
                    "input": str(inputs[index]),
                    "expected_output": str(expected_outputs[index]) if expected_outputs[index] else None,
                    "api_request_error": f"Internal execution error: {exc}",
                    "status": "internal_error",
                }

        # On short-circuit, drop the queued-but-unstarted cases immediately
        # (cancel_futures). Cases already inside a sandbox call finish and bail
        # via stop_event when the pool shuts down at the with-block exit.
        if stop_event is not None and stop_event.is_set():
            executor.shutdown(wait=False, cancel_futures=True)

    # Post-processing for compile errors
    if first_compile_error_index != -1:
        logger.warning(f"Compile error detected in case {first_compile_error_index}. Marking subsequent cases as compile errors.")
        for i in range(first_compile_error_index + 1, num_cases):
            # Only update if not already processed (though it should be None or have a result)
            if results[i] != -4:  # Avoid overwriting if it somehow already got -4
                results[i] = -4
                # Update or create metadata for skipped cases due to compile error
                if metadata_list[i] is None:  # If future failed before returning metadata
                    metadata_list[i] = {
                        "case_index": i,
                        "input": str(inputs[i]),
                        "expected_output": str(expected_outputs[i]) if expected_outputs[i] else None,
                        "api_request_error": None,
                        "status": "compile_error_skipped",  # Indicate skipped due to prior compile error
                    }
                else:  # If future completed but result is overridden
                    metadata_list[i]["status"] = "compile_error_skipped"

    logger.info(f"Correctness check finished. Results: {results}")
    return results, metadata_list
