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
"""Basic happy-path tests for the Rust sandbox runner.

Covers println! output, timeout enforcement, assert_eq! assertion
errors, compilation errors, and stdin delivery.
"""

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

def test_rust_print():
    """println! should compile, run, and produce expected stdout."""
    request = RunCodeRequest(language='rust',
                             code='''
    fn main() {
        println!("123");
    }
    ''',
                             run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result)
    assert result.status == RunStatus.Success
    assert result.compile_result.status == CommandRunStatus.Finished
    assert result.run_result.stdout.strip() == '123'

def test_rust_timeout():
    """thread::sleep exceeding the run_timeout must be killed and reported as TimeLimitExceeded."""
    request = RunCodeRequest(language='rust',
                             code='''
    use std::{thread, time};

    fn main() {
        let sleep_duration = time::Duration::from_millis(200);
        thread::sleep(sleep_duration);
    }
    ''',
                             run_timeout=0.1,
                             compile_timeout=20)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded

def test_rust_assertion_error():
    """assert_eq! with mismatched values should produce an assertion panic in stderr."""
    request = RunCodeRequest(language='rust', code='''
    fn main() {
        assert_eq!(1, 2);
    }
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.Finished
    assert "assertion" in result.run_result.stderr

def test_rust_compile_error():
    """A type mismatch should fail at compilation with a non-zero return code and no run_result."""
    request = RunCodeRequest(language='rust',
                             code='''
    fn main() {
        let x: u32 = "Hello, world!";
        println!("{}", x);
    }
    ''')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.compile_result.status == CommandRunStatus.Finished
    assert result.compile_result.return_code != 0
    assert result.run_result is None

def test_rust_stdin():
    """Stdin data should be delivered to the Rust program and readable via io::stdin."""
    request = RunCodeRequest(language='rust',
                             code='''
    use std::io;

    fn main() {
        let mut input = String::new();
        io::stdin().read_line(&mut input).expect("Failed to read line");

        let num: i32 = input.trim().parse().expect("Invalid input");
        println!("{}", num);
    }
    ''',
                             stdin='65535')
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert result.run_result.status == CommandRunStatus.Finished
    assert result.run_result.stdout == '65535\n'
