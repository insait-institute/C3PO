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
"""Basic happy-path tests for the Scala sandbox runner.

Covers println output, timeout enforcement, and Scala assert() for both
passing and failing cases using both ``object ... def main`` and
``object ... extends App`` entry-point styles.
All tests are marked ``pytest.mark.minor``.
"""

import pytest

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

@pytest.mark.minor
def test_scala_print():
    """Scala println should compile, run, and produce expected stdout."""
    request = RunCodeRequest(language='scala',
                             code='''
object HelloWorld {
    def main(args: Array[String]): Unit = {
        println("Hello, World!")
    }
}
    ''',
                             compile_timeout=20,
                             run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result)
    assert result.status == RunStatus.Success
    assert "Hello, World!" in result.run_result.stdout.strip()

@pytest.mark.minor
def test_scala_timeout():
    """Thread.sleep exceeding the run_timeout must be killed and reported as TimeLimitExceeded."""
    request = RunCodeRequest(language='scala',
                             code='''
object HelloWorld {
    def main(args: Array[String]): Unit = {
        Thread.sleep(5 * 1000)
        println("Hello, World!")
    }
}
    ''',
                             compile_timeout=20,
                             run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded

@pytest.mark.minor
def test_scala_assertion_success():
    """Passing Scala assert() should succeed for both main-method and App-trait styles."""
    request = RunCodeRequest(language='scala',
                             code='''
    object HelloWorld {
        def main(args: Array[String]): Unit = {
            assert((0l) == (0l));
        }
    }
        ''',
                             compile_timeout=20)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished

    request = RunCodeRequest(language='scala',
                             code='''
    object Main extends App {
        def foo(): Unit = {
            assert((0l) == (0l));
        }

        foo()
    }
        ''',
                             compile_timeout=20)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished

@pytest.mark.minor
def test_scala_assertion_error():
    """Failing Scala assert() should result in Failed status with 'assertion failed' in stderr."""
    request = RunCodeRequest(language='scala',
                             code='''
object HelloWorld {
    def main(args: Array[String]): Unit = {
        assert((1l) == (0l));
    }
}
    ''',
                             compile_timeout=20)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
    assert "assertion failed" in result.run_result.stderr

    request = RunCodeRequest(language='scala',
                             code='''
    object Main extends App {
        def foo(): Unit = {
            assert((1l) == (0l));
        }

        foo()
    }
        ''',
                             compile_timeout=20)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
