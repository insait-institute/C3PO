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
"""Basic happy-path tests for the Lean 4 (with Mathlib) sandbox runner.

Covers a valid Lean proof, a proof using ``sorry`` (which succeeds with
a warning), and a proof with a missing tactic step that causes a type
error.  All tests are marked ``pytest.mark.lean``.
"""

import pytest

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

@pytest.mark.lean
def test_lean_pass():
    """A valid Lean proof and a sorry-based proof should both succeed.

    The first proof is complete (using linarith); the second uses ``sorry``
    and should succeed but emit a 'declaration uses sorry' warning in stdout.
    """
    request = RunCodeRequest(language='lean',
                             code='''
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

theorem amc12_2000_p5
  (x p : ℝ)
  (h₀ : x < 2)
  (h₁ : abs (x - 2) = p) :
  x - p = 2 - 2 * p := by
  have : abs (x - 2) = -(x - 2) := by
    apply abs_of_neg
    linarith
  rw [h₁] at this
  linarith
                             ''',
                             run_timeout=30)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result.model_dump_json(indent=2))
    assert result.status == RunStatus.Success

    request = RunCodeRequest(language='lean',
                             code='''
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

theorem amc12_2000_p5
  (x p : ℝ)
  (h₀ : x < 2)
  (h₁ : abs (x - 2) = p) :
  x - p = 2 - 2 * p := by
  sorry
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result.model_dump_json(indent=2))
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished
    assert "declaration uses 'sorry'" in result.run_result.stdout

@pytest.mark.lean
def test_lean_error():
    """A Lean proof with a missing tactic step should fail type-checking."""
    request = RunCodeRequest(language='lean',
                             code='''
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith

theorem amc12_2000_p5
  (x p : ℝ)
  (h₀ : x < 2)
  (h₁ : abs (x - 2) = p) :
  x - p = 2 - 2 * p := by
  have : abs (x - 2) = -(x - 2) := by
    apply abs_of_neg
    -- linarith
  rw [h₁] at this
  linarith
    ''',
                             run_timeout=30)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result.model_dump_json(indent=2))
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
