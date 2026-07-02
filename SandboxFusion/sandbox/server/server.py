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

"""FastAPI application entry point for the SandboxFusion server.

Creates the main FastAPI application, conditionally mounts the static
documentation site (docs/build/), registers API routers for sandbox
execution and evaluation submission, and installs a global exception
handler that returns JSON-formatted tracebacks.
"""

import os
import traceback

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from sandbox.server.sandbox_api import sandbox_router
from sandbox.server.submit_api import submit_router
from sandbox.utils.logging import configure_logging

configure_logging()
logger = structlog.stdlib.get_logger()

app = FastAPI()


@app.on_event("startup")
async def _sweep_orphaned_sandboxes():
    """Clean orphaned sandbox resources from prior server crashes on startup."""
    from sandbox.runners.isolation import cleanup_orphaned_sandboxes
    cleaned = cleanup_orphaned_sandboxes()
    if cleaned:
        logger.warning('startup sweep cleaned orphaned sandboxes', count=cleaned)

docs_dir = os.path.abspath(os.path.join(__file__, '../../../docs/build'))
if os.path.isdir(docs_dir):
    app.mount('/SandboxFusion', StaticFiles(directory=docs_dir, html=True), name='doc-site')


@app.get("/", response_class=HTMLResponse)
async def root():
    """Redirect to the ``/SandboxFusion`` documentation site.

    Returns an HTML page whose inline JavaScript computes the correct
    documentation URL relative to the current origin.  For deployments on
    Hugging Face Spaces (``hf.space`` / ``huggingface.co``), the redirect
    URL is rewritten to use HTTPS to avoid mixed-content issues.
    """
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Redirecting...</title>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            const currentPath = window.location.pathname;
            const newPath = currentPath.endsWith('/') ? `${currentPath}SandboxFusion` : `${currentPath}/SandboxFusion`;
            let newUrl = `${window.location.origin}${newPath}`;
            if (newUrl.includes('hf.space') || newUrl.includes('huggingface.co')) {
                newUrl = newUrl.replace(/http:\\/\\//g, 'https://');
            }
            window.location.href = newUrl;
        });
    </script>
</head>
<body>
    <p>Redirecting to the documentation...</p>
</body>
</html>
    '''


@app.exception_handler(Exception)
async def base_exception_handler(request: Request, exc: Exception):
    """Global fallback exception handler.

    Catches any unhandled exception, formats a full Python traceback, and
    returns it as a ``500`` JSON response with ``detail`` (the exception
    message) and ``traceback`` (a list of traceback lines) fields.
    """
    return JSONResponse(status_code=500, content={'detail': str(exc), 'traceback': traceback.format_exc().split('\n')})


app.include_router(sandbox_router, tags=['sandbox'])
app.include_router(submit_router, tags=['eval'])


@app.get("/v1/ping")
async def index():
    """Health-check endpoint.

    Returns the literal string ``"pong"`` to confirm the server is
    running and reachable.
    """
    return "pong"
