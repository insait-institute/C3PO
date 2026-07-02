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

"""Runtime configuration loader for the sandbox service.

This module defines :class:`RunConfig`, a pydantic model that is populated
from a YAML configuration file at startup.  The YAML file is chosen by the
``SANDBOX_CONFIG`` environment variable (default ``"local"``), and the
corresponding ``<name>.yaml`` is resolved relative to this module's directory.

``RunConfig`` follows the **singleton pattern**: call
:meth:`RunConfig.get_instance_sync` to obtain the shared instance.  Direct
construction via ``RunConfig()`` is supported but discouraged outside tests.
"""

import os
from typing import Literal, Optional

import structlog
import yaml
from pydantic import BaseModel

logger = structlog.stdlib.get_logger()


class RunConfig(BaseModel):
    """Top-level runtime configuration singleton.

    Sections
    --------
    sandbox : SandboxConfig
        Controls the code-execution sandbox (isolation mode, concurrency,
        Docker image).
    eval : EvalConfig
        Controls parallel evaluation of test cases.
    common : Common
        Miscellaneous settings (e.g. logging).

    The configuration is loaded once from a YAML file and cached as a
    class-level singleton accessible via :meth:`get_instance_sync`.
    """

    class SandboxConfig(BaseModel):
        """Sandbox execution environment settings.

        Attributes
        ----------
        isolation : ``"lite"`` | ``"full"`` | ``"bindroot"``
            ``"lite"`` -- handcrafted overlayfs + chroot + cgroups isolation,
            fast (< 100 ms overhead).
            ``"full"`` -- Docker container isolation with resource limits.
            ``"bindroot"`` -- recursive bind-mount of ``/`` + per-exec
            tmpfs scratch + chroot.  Apptainer-compatible alternative to
            ``"lite"`` for hosts where overlay-on-rootfs is unavailable.
        max_concurrency : int
            Maximum number of sandbox instances that may run in parallel.
            Set to ``0`` to disable the internal concurrency limiter (useful
            when concurrency is managed externally, e.g. by pytest-xdist).
        queue_timeout : float
            Admission-control budget, in seconds, for how long a request may
            wait for a ``max_concurrency`` slot before the server gives up and
            returns HTTP 503.  ``0`` (the default) disables admission control
            and requests queue indefinitely (original behaviour).  Set this
            below the client's read timeout minus its run/compile budget so a
            saturated request gets a fast, retryable 503 instead of sitting on
            the connection until the client read-times-out.  See
            ``documentation/apptainer-migration.md`` section 6.4.
        docker_image : str
            Docker image used when ``isolation`` is ``"full"``.
            Defaults to ``"ineil77/sandbox-fusion-server:25042026-2"``.
        default_memory_limit_mb : int
            Default memory limit in megabytes for each sandbox execution.
            Overridden by per-request ``memory_limit_MB`` when > 0.
        default_cpu_limit : float
            Default CPU core limit for each sandbox execution.
            In lite mode this sets the CFS quota; in full mode it maps
            to ``docker run --cpus``.
        docker_startup_overhead : float
            Extra seconds added to compile and run timeouts in full
            (Docker) isolation mode to account for container startup
            latency.  Has no effect in lite mode.
        """
        isolation: Literal['lite', 'full', 'bindroot']
        max_concurrency: int
        queue_timeout: float = 0
        docker_image: str = 'ineil77/sandbox-fusion-server:25042026-2'
        default_memory_limit_mb: int = 8192
        default_cpu_limit: float = 2
        docker_startup_overhead: float = 10

    class EvalConfig(BaseModel):
        """Evaluation runner settings.

        Attributes
        ----------
        max_runner_concurrency : int
            Maximum number of test cases evaluated concurrently by the
            evaluation runner.  ``0`` means no limit.
        """
        max_runner_concurrency: int = 0

    class Common(BaseModel):
        """Miscellaneous / cross-cutting settings.

        Attributes
        ----------
        logging_color : bool
            When ``True``, structlog output includes ANSI colour codes.
        """
        logging_color: bool

    sandbox: SandboxConfig
    eval: EvalConfig = EvalConfig()
    common: Common

    def __init__(self):
        """Load configuration from the YAML file indicated by ``SANDBOX_CONFIG``.

        The environment variable ``SANDBOX_CONFIG`` (default ``"local"``)
        selects the ``<name>.yaml`` file located alongside this module.  The
        file is read, parsed with :func:`yaml.safe_load`, and the resulting
        dict is forwarded to the pydantic ``BaseModel`` constructor.
        """
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), f'{os.getenv("SANDBOX_CONFIG", "local")}.yaml'))
        with open(config_path) as f:
            data = yaml.safe_load(f)
        # SANDBOX_MAX_CONCURRENCY overrides sandbox.max_concurrency so one
        # yaml serves allocations of different core counts.  Keep it
        # <= cores / default_cpu_limit, or overflow execs run unpinned.
        max_concurrency = os.getenv("SANDBOX_MAX_CONCURRENCY")
        if max_concurrency:
            data["sandbox"]["max_concurrency"] = int(max_concurrency)
        # SANDBOX_QUEUE_TIMEOUT overrides sandbox.queue_timeout the same way,
        # so admission control can be enabled/tuned per allocation without
        # editing the yaml.
        queue_timeout = os.getenv("SANDBOX_QUEUE_TIMEOUT")
        if queue_timeout:
            data["sandbox"]["queue_timeout"] = float(queue_timeout)
        super().__init__(**data)

    # singleton logic
    _instance: Optional['RunConfig'] = None

    @classmethod
    def get_instance_sync(cls, *args, **kwargs) -> 'RunConfig':
        """Return the cached singleton, creating it on first call.

        If no instance exists yet, one is constructed (which triggers YAML
        loading).  Subsequent calls return the same object.

        Raises
        ------
        AssertionError
            If the class defines an ``async_init`` method -- in that case
            the caller should use the async counterpart instead.
        """
        if not cls.__private_attributes__['_instance'].default:
            self = cls(*args, **kwargs)
            assert not hasattr(
                self, 'async_init'), f'class {cls.__name__} has async_init function, init it with get_instance_async.'
            cls.__private_attributes__['_instance'].default = self
            logger.debug('singleton class initialized', name=cls.__name__)
        return cls.__private_attributes__['_instance'].default
