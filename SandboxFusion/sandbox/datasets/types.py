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

"""Pydantic models for the evaluation pipeline.

This module defines every data-transfer object used when submitting code to
be evaluated, capturing individual test-case outcomes, and reporting the
final evaluation result.

It also re-exports frequently-used types from :mod:`sandbox.runners.types`
and :mod:`sandbox.server.sandbox_api` so that downstream code can import
everything from a single location.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from sandbox.runners.types import CommandRunResult, CommandRunStatus, Language  # nopycln: import
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus  # nopycln: import


# ---------------------------------------------------------------------------
# OJ / evaluation related models
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single chat message.

    Attributes
    ----------
    role : str
        The speaker role (e.g. ``"user"``, ``"assistant"``, ``"system"``).
    content : str
        The textual content of the message.
    """
    role: str
    content: str


class Prompt(BaseModel):
    """A problem prompt to be sent to a model for code generation.

    Attributes
    ----------
    id : int | str
        Unique identifier for the problem.
    prompt : str | List[Message]
        Either a plain-text prompt string or a list of chat :class:`Message`
        objects that represent a multi-turn conversation.
    labels : Dict[str, Any]
        Arbitrary metadata labels associated with the prompt (e.g. difficulty,
        tags, source dataset).  Defaults to an empty dict.
    """
    id: int | str
    prompt: str | List[Message]
    labels: Dict[str, Any] = {}


class GeneralStdioTest(BaseModel):
    """A single stdin/stdout test case for standard-IO problems.

    Keys in the dictionaries are stream names: ``"stdin"`` / ``"stdout"``
    for the standard streams, or arbitrary file names for file-based I/O.

    Attributes
    ----------
    input : Dict[str, str]
        Mapping of stream/file name to the content that will be fed as input.
    output : Dict[str, str]
        Mapping of stream/file name to the expected output content.
    """
    input: Dict[str, str]
    output: Dict[str, str]


class TestConfig(BaseModel):
    """Configuration that governs how a submission is compiled, run, and judged.

    ``__test__`` is set to ``False`` so that pytest does not attempt to collect
    this class as a test suite.

    Attributes
    ----------
    language : Language | None
        Programming language of the submission.  ``None`` means auto-detect.
    locale : str | None
        Optional locale hint (e.g. ``"en"``, ``"zh"``).
    compile_timeout : float | None
        Maximum seconds allowed for compilation.  ``None`` uses the default.
    run_timeout : float | None
        Maximum seconds allowed for execution.  ``None`` uses the default.
    custom_extract_logic : str | None
        A snippet of Python code that calls ``submit_code_blocks(cbs)`` to
        override the default code-extraction heuristic.  ``cbs`` is a
        ``List[CodeBlock]`` where each ``CodeBlock`` has ``priority``,
        ``code``, and ``language`` fields.  Built-in priority levels:
        fenced = 30, incomplete fenced = 20, heuristic = 10.  Custom blocks
        typically use priority 40 to take precedence.
    extra : Dict[str, Any]
        Catch-all dictionary for dataset-specific configuration that does
        not fit into the standard fields.
    """
    __test__ = False
    language: Optional[Language] = None
    locale: Optional[str] = None
    compile_timeout: Optional[float] = None
    run_timeout: Optional[float] = None
    custom_extract_logic: Optional[str] = None
    extra: Dict[str, Any] = {}


class EvalTestCase(BaseModel):
    """Result of a single test case execution.

    Attributes
    ----------
    passed : bool
        Whether the test case passed (output matched expectations).
    exec_info : RunCodeResponse
        Low-level execution details (exit code, stdout, stderr, run status).
    test_info : Dict[str, Any] | None
        Optional extra information produced by the judge (e.g. diff details,
        partial scores).
    """
    passed: bool
    exec_info: RunCodeResponse
    test_info: Optional[Dict[str, Any]] = None


class EvalResult(BaseModel):
    """Aggregated evaluation result for a single problem submission.

    Attributes
    ----------
    id : int | str
        Identifier of the problem that was evaluated.
    accepted : bool
        ``True`` if **all** test cases passed.
    extracted_code : str
        The source code extracted from the model's completion.
    full_code : str | None
        The complete code that was actually compiled/run (may include
        harness or wrapper code added by the runner).
    test_code : str | None
        The test/driver code used to evaluate the submission, if applicable.
    tests : List[EvalTestCase]
        Per-test-case results.
    extracted_type : str | None
        How the code was extracted from the completion.  One of
        ``"fenced"``, ``"incomplete_fenced"``, ``"heuristic"``, or
        ``"empty"``.
    extra : Dict | None
        Optional extra metadata produced during evaluation.
    """
    id: int | str
    accepted: bool
    extracted_code: str
    full_code: Optional[str] = None
    test_code: Optional[str] = None
    tests: List[EvalTestCase]
    extracted_type: Optional[Literal['fenced', 'incomplete_fenced', 'heuristic', 'empty']] = None
    extra: Optional[Dict] = None


class SubmitRequest(BaseModel):
    """Payload for the ``/submit`` evaluation endpoint.

    Attributes
    ----------
    id : int | str
        Problem identifier.
    completion : str
        Raw model completion from which code will be extracted.
    config : TestConfig
        Evaluation configuration (language, timeouts, extraction logic, etc.).
    test_cases : List[GeneralStdioTest]
        The test cases to run against the extracted code.
    """
    id: int | str
    completion: str
    config: TestConfig
    test_cases: List[GeneralStdioTest]
