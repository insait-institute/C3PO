"""SandboxFusion client configuration.

Holds the global ``SANDBOX_ENDPOINT`` variable used by both the synchronous
and asynchronous clients. The endpoint is read from the
``SANDBOX_FUSION_ENDPOINT`` environment variable at import time, defaulting
to ``http://localhost:8080`` when the variable is not set. It can be changed
at runtime via :func:`sandbox_fusion.client.set_endpoint`.
"""

import os

SANDBOX_ENDPOINT = os.environ.get('SANDBOX_FUSION_ENDPOINT', 'http://localhost:8080')
