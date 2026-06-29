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
"""Entry point for all language runners in the sandbox.

This module aggregates the ``MAJOR_RUNNERS`` and ``MINOR_RUNNERS`` dictionaries
into a single ``CODE_RUNNERS`` mapping that maps language identifier strings to
their corresponding async runner functions.  It also re-exports the core type
definitions (``CodeRunArgs``, ``CodeRunResult``, ``CommandRunResult``,
``CommandRunStatus``, and ``Language``) so that consumers can import everything
they need from ``sandbox.runners`` directly.
"""

from sandbox.runners.major import MAJOR_RUNNERS
from sandbox.runners.minor import MINOR_RUNNERS
from sandbox.runners.types import (  # nopycln: import
    CodeRunArgs, CodeRunResult, CommandRunResult, CommandRunStatus, Language,
)

CODE_RUNNERS = {
    **MAJOR_RUNNERS,
    **MINOR_RUNNERS,
}

__all__ = [
    'CODE_RUNNERS', 'CodeRunArgs', 'CodeRunResult', 'CommandRunResult', 'CommandRunStatus', 'Language'
]
