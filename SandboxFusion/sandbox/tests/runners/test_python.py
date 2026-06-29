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
"""Basic happy-path tests for the Python sandbox runner.

Covers fundamental execution scenarios: printing to stdout, timeout
enforcement, assertion errors, syntax errors, reading provided files,
stdin delivery, and fetching output files after execution.
"""

import base64
import os

import pytest

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

def test_python_print():
    """Simple print statement should succeed and produce the expected stdout."""
    request = RunCodeRequest(language='python', code='print(123)', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.stdout.strip() == '123'

def test_python_timeout():
    """A sleep exceeding the run_timeout must be killed and reported as TimeLimitExceeded."""
    request = RunCodeRequest(language='python', code='import time; time.sleep(0.2)', run_timeout=0.1)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded

def test_python_assertion_error():
    """A failing assert should produce AssertionError in stderr and a Failed status."""
    request = RunCodeRequest(language='python', code='assert 1 == 2', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
    assert 'AssertionError' in result.run_result.stderr

def test_python_syntax_error():
    """Invalid Python syntax should produce SyntaxError in stderr and a Failed status."""
    request = RunCodeRequest(language='python', code='int a = 1', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
    assert 'SyntaxError: invalid syntax' in result.run_result.stderr

def test_python_file_read():
    """A base64-encoded file provided in the files dict should be readable at the given path."""
    request = RunCodeRequest(language='python',
                             code='print(open("dir1/dir2/dir3/secret_flag").read())',
                             run_timeout=5,
                             files={'dir1/dir2/dir3/secret_flag': "ImhlbGxvLCB0aGlzIGlzIGEgdGVzdCI="})
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished
    assert 'hello, this is a test' in result.run_result.stdout

def test_python_stdin():
    """Stdin data should be delivered to the program and readable via input()."""
    request = RunCodeRequest(language='python', code='print(int(input()))', run_timeout=5, stdin='65535')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished
    assert result.run_result.stdout == '65535\n'

def test_python_fetch_files():
    """Files written during execution should be retrievable via fetch_files as base64."""
    request = RunCodeRequest(language='python',
                             code='open("a.txt", "w").write("secret")',
                             run_timeout=5,
                             fetch_files=['a.txt'])
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished
    assert base64.b64decode(result.files['a.txt'].encode()).decode() == 'secret'

@pytest.mark.skipif(
    os.environ.get('SANDBOX_ISOLATION_MODE') in ('full', 'bindroot'),
    reason='Full and bindroot modes only expose the working directory; absolute paths outside cwd are not retrievable')
def test_python_fetch_files_absolute_path():
    """Files written to absolute paths should be retrievable in lite mode via overlayfs."""
    request = RunCodeRequest(language='python',
                             code='open("/mnt/b", "w").write("sauce")',
                             run_timeout=5,
                             fetch_files=['/mnt/b'])
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished
    assert base64.b64decode(result.files['/mnt/b'].encode()).decode() == 'sauce'
