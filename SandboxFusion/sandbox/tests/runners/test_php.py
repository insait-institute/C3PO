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
"""Basic happy-path tests for the PHP sandbox runner.

Covers echo output, timeout enforcement, E_USER_ERROR triggering, and
stdin delivery via readline.
"""

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

def test_php_print():
    """PHP echo should produce expected stdout."""
    request = RunCodeRequest(language='php', code='''
    <?php
    echo "123";
    ?>
    ''', run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.stdout.strip() == '123'

def test_php_timeout():
    """sleep(2) exceeding the run_timeout must be killed and reported as TimeLimitExceeded."""
    request = RunCodeRequest(language='php',
                             code='''
    <?php
    sleep(2); // Sleep for 2 seconds
    ?>
    ''',
                             run_timeout=0.1)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded

def test_php_error():
    """trigger_error with E_USER_ERROR should produce the error message and a Failed status."""
    request = RunCodeRequest(language='php',
                             code='''
    <?php
    trigger_error("Custom error", E_USER_ERROR);
    ?>
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert "Custom error" in result.run_result.stderr + result.run_result.stdout

def test_php_stdin():
    """Stdin data should be delivered to the PHP process and readable via readline."""
    request = RunCodeRequest(language='php',
                             code='''
    <?php
        $input = readline();
        echo intval($input);
    ?>
                             ''',
                             run_timeout=5,
                             stdin='65535')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished
    assert '65535' in result.run_result.stdout
