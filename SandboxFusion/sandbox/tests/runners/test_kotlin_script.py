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
"""Basic happy-path tests for the Kotlin script sandbox runner.

Covers println output, timeout enforcement, and exception-based
assertions for both passing and failing cases.
All tests are marked ``pytest.mark.minor``.
"""

import pytest

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

@pytest.mark.minor
def test_kotlin_script_print():
    """println should produce expected stdout in Kotlin script mode."""
    request = RunCodeRequest(language='kotlin_script', code='''
println("Hello, World!")
    ''', run_timeout=30)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert "Hello, World!" in result.run_result.stdout.strip()

@pytest.mark.minor
def test_kotlin_script_timeout():
    """Thread.sleep exceeding the run_timeout must be killed and reported as TimeLimitExceeded."""
    request = RunCodeRequest(language='kotlin_script',
                             code='''
fun main() {
    println("Starting...")
    
    // 让程序暂停 2 秒
    Thread.sleep(2000)
    
    println("Finished!")
}

main()
    ''',
                             run_timeout=1)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded

@pytest.mark.minor
def test_kotlin_script_assertion_success():
    """A matching expected value should not throw and should result in Success status."""
    request = RunCodeRequest(language='kotlin_script',
                             code='''
fun minCost() : Int {
    return 0
}

fun main() {
    var x : Int = minCost();
    var y : Int = 0;
    if (x != y) {
        throw Exception("Exception -- test case did not pass. x = " + x)
    }
}

main()
    ''',
                             run_timeout=40)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished

@pytest.mark.minor
def test_kotlin_script_assertion_error():
    """A mismatched expected value should throw an Exception and result in Failed status."""
    request = RunCodeRequest(language='kotlin_script',
                             code='''
fun minCost() : Int {
    return 0
}

fun main() {
    var x : Int = minCost();
    var y : Int = 1;
    if (x != y) {
        throw Exception("Exception -- test case did not pass. x = " + x)
    }
}

main()
    ''',
                             run_timeout=20)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result)
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
    assert "java.lang.Exception" in result.run_result.stderr
