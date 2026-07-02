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
"""Unit tests for the sandbox_fusion stdin/stdout output matcher.

Guards the competitive-programming-grade leniency (per-line strip + float
tolerance, layout-preserving) that replaced the old bare ``rstrip("\\n")``
exact match -- the strict match systematically scored correct solutions wrong
on formatting/float differences and flattened the GRPO reward signal.
"""

import pytest

from verl.utils.reward_score.sandbox_fusion.utils import _outputs_match


@pytest.mark.parametrize(
    "actual,expected,want",
    [
        # per-line trailing whitespace must not matter
        ("3\n4", "3 \n4\n", True),
        ("YES", "YES\n", True),
        ("42\n", "42", True),
        # float precision within tolerance passes (mirrors np.allclose)
        ("0.500001", "0.5\n", True),
        ("0.5\n0.25\n", "0.500000\n0.250000\n", True),
        ("3", "3.0000001", True),
        ("-0.0", "0.0", True),
        # genuine differences still fail
        ("YES", "NO", False),
        ("3", "3.5", False),
        # layout differences must fail (chosen leniency: layout-preserving)
        ("1\n2\n3", "1 2 3\n", False),
        ("1\n2\n", "2\n1\n", False),
        # exact / empty
        ("1 2 3\n", "1 2 3", True),
        ("", "", True),
        ("1000000007", "1000000007", True),
    ],
)
def test_outputs_match(actual, expected, want):
    assert _outputs_match(actual, expected) is want
