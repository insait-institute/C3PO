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
"""Basic happy-path tests for the Go test (go_test) sandbox runner.

Covers passing tests, failing tests (with testify assertions), and
timeout enforcement when running ``go test`` style code.
"""

from sandbox.runners import CommandRunStatus
from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

def test_golang_test_pass():
    """A correct HasCloseElements implementation should pass all testify assertions."""
    request = RunCodeRequest(language='go_test',
                             code='''
    package main

    import (
        "math"
        "testing"
        "github.com/stretchr/testify/assert"
    )

    // Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    // >>> HasCloseElements([]float64{1.0, 2.0, 3.0}, 0.5)
    // false
    // >>> HasCloseElements([]float64{1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3)
    // true
    func HasCloseElements(numbers []float64, threshold float64) bool {
        for i := 0; i < len(numbers); i++ {
            for j := i + 1; j < len(numbers); j++ {
                var distance float64 = math.Abs(numbers[i] - numbers[j])
                if distance < threshold {
                    return true
                }
            }
        }
        return false
    }

    func TestHasCloseElements(t *testing.T) {
        assert := assert.New(t)
        assert.Equal(false, HasCloseElements([]float64{1.0, 2.0, 3.0}, 0.5))
        assert.Equal(true, HasCloseElements([]float64{1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3))
    }
    ''',
                             run_timeout=120,
                             compile_timeout=30)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Success
    assert 'ok' in result.run_result.stdout.strip()

def test_golang_test_fail():
    """An incorrect implementation (inverted logic) should fail with 'Not equal' in stdout."""
    request = RunCodeRequest(language='go_test',
                             code='''
    package main

    import (
        "math"
        "testing"
        "github.com/stretchr/testify/assert"
    )

    // Check if in given list of numbers, are any two numbers closer to each other than given threshold.
    // >>> HasCloseElements([]float64{1.0, 2.0, 3.0}, 0.5)
    // false
    // >>> HasCloseElements([]float64{1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3)
    // true
    func HasCloseElements(numbers []float64, threshold float64) bool {
        for i := 0; i < len(numbers); i++ {
            for j := i + 1; j < len(numbers); j++ {
                var distance float64 = math.Abs(numbers[i] - numbers[j])
                if distance < threshold {
                    return false
                }
            }
        }
        return true
    }

    func TestHasCloseElements(t *testing.T) {
        assert := assert.New(t)
        assert.Equal(false, HasCloseElements([]float64{1.0, 2.0, 3.0}, 0.5))
        assert.Equal(true, HasCloseElements([]float64{1.0, 2.8, 3.0, 4.0, 5.0, 2.0}, 0.3))
    }
    ''',
                             run_timeout=120,
                             compile_timeout=30)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result)
    assert result.status == RunStatus.Failed
    assert 'Not equal' in result.run_result.stdout.strip()

def test_golang_test_timeout():
    """A test that sleeps longer than run_timeout should be killed as TimeLimitExceeded."""
    request = RunCodeRequest(language='go_test',
                             code='''
    package main

    import (
        "time"
        "testing"
    )

    func TestTimeout(t *testing.T) {
        time.Sleep(200 * time.Millisecond)
    }
    ''',
                             run_timeout=0.19,
                             compile_timeout=30)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    assert result.status == RunStatus.Failed
    assert result.run_result.status == CommandRunStatus.TimeLimitExceeded
