"""HTTP test client for the SandboxFusion test suite.

Tests always run against a real server over HTTP.  The server URL is
read from ``SANDBOX_TEST_SERVER_URL`` (set automatically by conftest
when ``--sandbox-mode`` starts the container).
"""

import os

import httpx


def _make_client():
    url = os.environ.get('SANDBOX_TEST_SERVER_URL')
    if not url:
        raise RuntimeError(
            'SANDBOX_TEST_SERVER_URL is not set. '
            'Run tests with --sandbox-mode full or --sandbox-mode lite '
            '(optionally with --sandbox-backend docker|apptainer).')
    return httpx.Client(base_url=url, timeout=120)


client = _make_client()
