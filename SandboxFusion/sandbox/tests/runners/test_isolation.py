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
"""Tests for network isolation and namespace separation in the sandbox.

Verifies that sandboxed code can:

* Start an HTTP server on 127.0.0.1 or localhost and reach it from the
  same sandbox (SERVER_1 / SERVER_2 templates).
* Run two servers on the **same** port concurrently without conflicts,
  proving that each sandbox gets its own network namespace.
* Make outbound HTTP requests to external hosts (NET_1 template).

These tests are critical for confirming that the sandbox network
isolation layer works correctly and does not leak between concurrent
executions.
"""

import asyncio
import os
import random

import pytest

from sandbox.server.sandbox_api import RunCodeRequest, RunCodeResponse, RunStatus

from sandbox.tests.client import client

SERVER_1 = '''
import http.server
import socketserver
import sys
import threading
import requests

# Define the handler to serve the request
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Hello, this is a special string!')

# Function to start the server
def start_server():
    try:
        PORT = {port}
        handler = MyHandler
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            httpd.serve_forever()
    except Exception as e:
        print(e)
        sys.exit(1)

# Run the server in a separate thread
server_thread = threading.Thread(target=start_server)
server_thread.daemon = True
server_thread.start()

# Function to make a request to the server and check the response
def test_server_response():
    url = "http://127.0.0.1:{port}"
    response = requests.get(url)
    
    if response.status_code == 200 and "special string" in response.text:
        print("Test Passed: Correct response received.")
    else:
        print("Test Failed: Incorrect response received.")

# Give the server a moment to start
import time
time.sleep({wait_time})

# Test the server response
test_server_response()
'''

SERVER_2 = '''
import http.server
import socketserver
import sys
import threading
import requests

# Define the handler to serve the request
class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Hello, this is a special string!')

# Function to start the server
def start_server():
    try:
        PORT = {port}
        handler = MyHandler
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            httpd.serve_forever()
    except Exception as e:
        print(e)
        sys.exit(1)

# Run the server in a separate thread
server_thread = threading.Thread(target=start_server)
server_thread.daemon = True
server_thread.start()

# Function to make a request to the server and check the response
def test_server_response():
    url = "http://localhost:{port}"
    response = requests.get(url)
    
    if response.status_code == 200 and "special string" in response.text:
        print("Test Passed: Correct response received.")
    else:
        print("Test Failed: Incorrect response received.")

# Give the server a moment to start
import time
time.sleep({wait_time})

# Test the server response
test_server_response()
'''

NET_1 = '''
import requests

def test_network_access():
    urls = [
        'https://httpbin.org/get',
        'https://www.google.com',
        'https://www.sina.com.cn',
    ]
    last_error = None
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print("Network access successful.")
                return
            last_error = "status code: {}".format(response.status_code)
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = str(e)
            continue
    raise Exception("All endpoints failed. Last error: {}".format(last_error))

test_network_access()
'''

def test_isolation_network_server_127():
    """Verify that a Python HTTP server bound to 127.0.0.1 is reachable within the sandbox.

    Starts a simple HTTP server on a random high port using the 127.0.0.1
    address, then issues a GET request from the same process.  Confirms
    that the loopback interface works inside the isolated environment.
    """
    request = RunCodeRequest(language='python',
                             code=SERVER_1.format(port=random.randint(30000, 60000), wait_time=1),
                             run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result.model_dump_json(indent=2))
    assert result.status == RunStatus.Success
    assert 'Test Passed' in result.run_result.stdout

def test_isolation_network_server_localhost():
    """Verify that a Python HTTP server bound to localhost is reachable within the sandbox.

    Same as ``test_isolation_network_server_127`` but uses the ``localhost``
    hostname to confirm DNS / hosts resolution works inside the namespace.
    """
    request = RunCodeRequest(language='python',
                             code=SERVER_2.format(port=random.randint(30000, 60000), wait_time=1),
                             run_timeout=5)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result.model_dump_json(indent=2))
    assert result.status == RunStatus.Success
    assert 'Test Passed' in result.run_result.stdout

async def test_isolation_network_server_port_conflict():
    """Verify that two concurrent servers on the same port do not conflict.

    Launches two sandbox executions in parallel, each starting an HTTP server
    on the **same** random port.  Because each run operates in a separate
    network namespace, both must succeed without ``Address already in use``
    errors, proving proper namespace isolation.
    """
    request = RunCodeRequest(language='python',
                             code=SERVER_1.format(port=random.randint(30000, 60000), wait_time=2),
                             run_timeout=6)

    def post():
        return client.post('/run_code', json=request.model_dump())

    results = await asyncio.gather(asyncio.to_thread(post), asyncio.to_thread(post))
    for response in results:
        assert response.status_code == 200
        result = RunCodeResponse(**response.json())
        print(result.model_dump_json(indent=2))
        assert result.status == RunStatus.Success
        assert 'Test Passed' in result.run_result.stdout

@pytest.mark.skipif(
    os.environ.get('SANDBOX_ISOLATION_MODE') in ('full', 'bindroot'),
    reason='Full mode uses --network none and bindroot uses a loopback-only '
           'netns (unshare -n); both block egress traffic by design')
def test_isolation_network_external_access():
    """Verify that sandboxed code can make outbound HTTP requests to external hosts.

    Runs code that performs a GET request to an external website and checks
    for a 200 status code, confirming that the sandbox allows egress traffic.
    Only applicable in lite mode where network namespaces provide NAT bridging.
    """
    request = RunCodeRequest(language='python', code=NET_1, run_timeout=60)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result.model_dump_json(indent=2))
    assert result.status == RunStatus.Success


@pytest.mark.skipif(
    os.environ.get('SANDBOX_ISOLATION_MODE') not in ('full', 'bindroot'),
    reason='Lite mode provides NAT bridging, so egress is allowed there')
def test_isolation_network_external_blocked():
    """Verify that sandboxed code can NOT reach external hosts.

    Full mode blocks egress via ``--network none``; bindroot blocks it via a
    loopback-only network namespace.  NET_1 raises when every endpoint is
    unreachable, so the run must finish with a non-zero exit code.
    """
    request = RunCodeRequest(language='python', code=NET_1, run_timeout=60)
    response = client.post('/run_code', json=request.model_dump())
    assert response.status_code == 200
    result = RunCodeResponse(**response.json())
    print(result.model_dump_json(indent=2))
    assert result.status == RunStatus.Failed
    assert 'All endpoints failed' in result.run_result.stderr
