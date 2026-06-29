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
"""Basic happy-path tests for the Julia sandbox runner.

Covers println output, timeout enforcement, and Julia ``@testset`` /
``@test`` assertions for both passing and failing cases.
All tests are marked ``pytest.mark.minor``.
"""

import pytest

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

@pytest.mark.minor
def test_julia_print():
    """println should produce expected stdout."""
    request = RunCodeRequest(language='julia', code='''
println("Hello, World!")
    ''', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result)
    assert result.status == RunStatus.Success
    assert "Hello, World!" in result.run_result.stdout.strip()

@pytest.mark.minor
def test_julia_timeout():
    """Base.sleep exceeding the run_timeout must be killed and reported as TimeLimitExceeded."""
    request = RunCodeRequest(language='julia', code='''
Base.sleep(5)
    ''', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded

@pytest.mark.minor
def test_julia_assertion_success():
    """A passing @test assertion should result in Success status."""
    request = RunCodeRequest(language='julia', code='''
using Test
@testset begin
	@test(0 == 0)
end
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished

@pytest.mark.minor
def test_julia_assertion_error():
    """A failing @test assertion should result in Failed status with 'Test Failed' in stdout."""
    request = RunCodeRequest(language='julia', code='''
using Test
@testset begin
	@test(1 == 0)
end
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
    assert "Test Failed" in result.run_result.stdout
