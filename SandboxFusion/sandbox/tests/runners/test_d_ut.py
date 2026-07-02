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
"""Basic happy-path tests for the D (with unittest) sandbox runner.

Covers stdout output, timeout enforcement, and the built-in D
``unittest`` block for both passing and failing assertions.
All tests are marked ``pytest.mark.minor``.
"""

import pytest

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

@pytest.mark.minor
def test_D_ut_print():
    """writeln should produce expected stdout after compilation and execution."""
    request = RunCodeRequest(language='D_ut',
                             code='''
import std.stdio; 
 
void main(string[] args) { 
   writeln("Hello, World!"); 
}
    ''',
                             run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result)
    assert result.status == RunStatus.Success
    assert "Hello, World!" in result.run_result.stdout.strip()

@pytest.mark.minor
def test_D_ut_timeout():
    """A sleep exceeding the run_timeout must be killed and reported as TimeLimitExceeded."""
    request = RunCodeRequest(language='D_ut',
                             code='''
import core.thread;
import std.stdio;

void sleepSeconds() {
    Thread.sleep( dur!("seconds")( 5 ) );
}

void main() {
    sleepSeconds();
    // 5秒后执行下面的代码
    writeln("5 seconds have passed.");
}
    ''',
                             run_timeout=1)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded

@pytest.mark.minor
def test_D_ut_assertion_success():
    """A passing D unittest assertion should result in Success status."""
    request = RunCodeRequest(language='D_ut', code='''
unittest
{
    assert(0 == 0);
}
void main(){}
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished

@pytest.mark.minor
def test_D_ut_assertion_error():
    """A failing D unittest assertion should result in Failed status."""
    request = RunCodeRequest(language='D_ut', code='''
unittest
{
    assert(0 == 1);
}
void main(){}
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
