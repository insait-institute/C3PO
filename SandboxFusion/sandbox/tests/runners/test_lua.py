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
"""Basic happy-path tests for the Lua sandbox runner.

Covers print output, timeout enforcement, and LuaUnit assertions for
both passing and failing cases.
All tests are marked ``pytest.mark.minor``.
"""

import pytest

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

@pytest.mark.minor
def test_lua_print():
    """Lua print should produce expected stdout."""
    request = RunCodeRequest(language='lua', code='''
print("Hello, World!")
    ''', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result)
    assert result.status == RunStatus.Success
    assert result.run_result.stdout.strip() == "Hello, World!"

@pytest.mark.minor
def test_lua_timeout():
    """os.execute('sleep') exceeding the run_timeout must be killed as TimeLimitExceeded."""
    request = RunCodeRequest(language='lua', code='''
os.execute("sleep 3")
    ''', run_timeout=1)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded

@pytest.mark.minor
def test_lua_assertion_success():
    """A passing LuaUnit assertEquals should result in Success status."""
    request = RunCodeRequest(language='lua', code='''
lu = require('luaunit')
lu.assertEquals(0, 0)
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished

@pytest.mark.minor
def test_lua_assertion_error():
    """A failing LuaUnit assertEquals should result in Failed status with error in stderr."""
    request = RunCodeRequest(language='lua', code='''
lu = require('luaunit')
lu.assertEquals(1, 2)
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
    assert "LuaUnit test FAILURE" in result.run_result.stderr
