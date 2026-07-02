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
"""Basic happy-path tests for the Swift sandbox runner.

Covers print output, timeout enforcement, assert() for both passing and
failing cases, and compilation errors.
"""

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

def test_swift_print():
    """Swift print should compile, run, and produce expected stdout."""
    request = RunCodeRequest(language='swift',
                             code='''
var myString = "Hello, World!"
 
print(myString)
    ''',
                             run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.compile_result.status == CommandRunStatus.Finished
    assert "Hello, World!" in result.run_result.stdout.strip()

def test_swift_timeout():
    """DispatchQueue delay exceeding the run_timeout must be killed as TimeLimitExceeded."""
    request = RunCodeRequest(language='swift',
                             code='''
import Foundation

print("Start")

DispatchQueue.main.asyncAfter(deadline: .now() + 10) {
    print("After 10 seconds")
    exit(0)
}

RunLoop.main.run()
    ''',
                             run_timeout=0.5,
                             compile_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded

def test_swift_assertion_success():
    """Passing Swift assert() calls with correct strlen results should succeed."""
    request = RunCodeRequest(language='swift',
                             code='''
import Foundation

/// Return length of given string
/// >>> strlen(string: "")
/// 0
/// >>> strlen(string: "abc")
/// 3
func strlen(string: String) -> Int {
    return string.count
}

func ==(left: [(Int, Int)], right: [(Int, Int)]) -> Bool {
    if left.count != right.count {
        return false
    }
    for (l, r) in zip(left, right) {
        if l != r {
            return false
        }
    }
    return true
}

assert(strlen(string: "") == 0)
assert(strlen(string: "x") == 1)
assert(strlen(string: "asdasnakj") == 9)
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished

def test_swift_assertion_error():
    """A failing Swift assert() (wrong expected length) should result in Failed status."""
    request = RunCodeRequest(language='swift',
                             code='''
import Foundation

/// Return length of given string
/// >>> strlen(string: "")
/// 0
/// >>> strlen(string: "abc")
/// 3
func strlen(string: String) -> Int {
    return string.count
}

func ==(left: [(Int, Int)], right: [(Int, Int)]) -> Bool {
    if left.count != right.count {
        return false
    }
    for (l, r) in zip(left, right) {
        if l != r {
            return false
        }
    }
    return true
}

assert(strlen(string: "") == 1)
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished

def test_swift_compile_error():
    """A missing closing parenthesis should fail compilation with a non-zero return code."""
    request = RunCodeRequest(language='swift', code='''
var myString = "Hello, World!"

print(myString
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.compile_result.status == CommandRunStatus.Finished
    assert result.compile_result.return_code != 0
    assert result.run_result is None
