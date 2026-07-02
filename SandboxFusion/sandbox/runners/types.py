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
"""Core type definitions shared across all sandbox runners.

This module defines the data models and enumerations used to describe runner
inputs, outputs, and execution statuses.  It also declares the ``Language``
literal type enumerating all 27 supported language identifiers and a
convenience list of languages that require a compilation step.
"""

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel


class CommandRunStatus(str, Enum):
    """Possible terminal statuses for a single shell command execution.

    Members:
        Finished: The command exited normally (check ``return_code`` for success/failure).
        Error: An unexpected exception occurred while trying to run the command.
        TimeLimitExceeded: The command was killed because it exceeded its timeout.
    """
    Finished = 'Finished'
    Error = 'Error'
    TimeLimitExceeded = 'TimeLimitExceeded'


class CommandRunResult(BaseModel):
    """Result of executing a single shell command.

    Attributes:
        status: Terminal status of the command.
        execution_time: Wall-clock seconds elapsed, or ``None`` on error.
        return_code: Process exit code, or ``None`` if the process did not finish.
        stdout: Captured standard output (may be ``None``).
        stderr: Captured standard error (may be ``None``).
    """
    status: CommandRunStatus
    execution_time: Optional[float] = None
    return_code: Optional[int] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class CodeRunArgs(BaseModel):
    """Input arguments accepted by every language runner function.

    Attributes:
        code: Source code to compile and/or execute.
        files: Mapping of relative file paths to base64-encoded content that
            should be restored into the working directory before execution.
        compile_timeout: Maximum seconds allowed for the compilation step.
        run_timeout: Maximum seconds allowed for the execution step.
        memory_limit_MB: Memory limit in megabytes (``-1`` means use the
            config default ``sandbox.default_memory_limit_mb``, typically 8192 MB).
        stdin: Optional string to feed to the program's standard input.
        fetch_files: List of relative file paths whose contents (base64-encoded)
            should be collected from the sandbox after execution and included
            in the result.
    """
    code: str
    files: Dict[str, Optional[str]] = {}
    compile_timeout: float = 10
    run_timeout: float = 10
    memory_limit_MB: int = -1
    stdin: Optional[str] = None
    fetch_files: List[str] = []


class CodeRunResult(BaseModel):
    """Output returned by every language runner function.

    Attributes:
        compile_result: Result of the compilation step, or ``None`` if the
            language does not require compilation.
        run_result: Result of the execution step, or ``None`` if compilation
            failed and execution was skipped.
        files: Mapping of requested file paths to their base64-encoded contents
            retrieved from the sandbox after execution.
    """
    compile_result: Optional[CommandRunResult] = None
    run_result: Optional[CommandRunResult] = None
    files: Dict[str, str] = {}


Language = Literal['python', 'cpp', 'nodejs', 'go', 'go_test', 'java', 'php', 'csharp', 'bash', 'typescript', 'sql',
                   'rust', 'lua', 'R', 'perl', 'D_ut', 'ruby', 'scala', 'julia', 'pytest', 'junit',
                   'kotlin_script', 'jest', 'verilog', 'lean', 'swift', 'racket']
"""Literal type representing all 27 supported language identifiers."""

compile_languages: List[Language] = ['cpp', 'go', 'java']
"""Languages that require a separate compilation step before execution."""
