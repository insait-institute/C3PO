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

"""Structured logging configuration using structlog.

Configures the structlog processing pipeline for the sandbox application,
including stdout console rendering (with optional color), optional JSON file
output for tracing, and silencing of noisy third-party loggers.
"""

import logging
import os
import sys

import structlog

from sandbox.configs.run_config import RunConfig

config = RunConfig.get_instance_sync()

# Root log level, overridable via the SANDBOX_LOG_LEVEL env var (e.g. DEBUG,
# INFO, WARNING, ERROR, CRITICAL). Defaults to INFO to avoid logging every
# code-execution request, which is extremely verbose under RL rollouts. Set it
# to OFF/NONE/DISABLE (or any level above CRITICAL) to silence all logging.
_level_name = os.environ.get('SANDBOX_LOG_LEVEL', 'INFO').upper()
if _level_name in ('OFF', 'NONE', 'DISABLE', 'DISABLED', 'SILENT'):
    LOG_LEVEL = logging.CRITICAL + 1
else:
    LOG_LEVEL = getattr(logging, _level_name, logging.INFO)


def configure_logging(trace_file=None):
    """Set up the structlog logging pipeline and handlers.

    Configures structlog with a standard processing chain (level filtering,
    logger name, log level, positional args formatting, timestamps, stack
    info, and exception formatting). Sets up:

    - A stdout ``StreamHandler`` with ``ConsoleRenderer`` (color controlled
      by ``RunConfig.common.logging_color``).
    - An optional JSON ``FileHandler`` when ``trace_file`` is provided, useful
      for machine-readable trace output.

    Also silences noisy loggers: ``aiosqlite``, ``databases``, and
    ``uvicorn.access``.

    Args:
        trace_file: Optional file path for JSON-formatted trace output. If
            ``None``, only stdout logging is configured.
    """

    def filter_keys(_, __, event_dict):
        event_dict.pop('_from_structlog', None)
        event_dict.pop('_record', None)
        return event_dict

    structlog.configure(processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
    ],
                        logger_factory=structlog.stdlib.LoggerFactory(),
                        context_class=dict)

    handlers = []

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(LOG_LEVEL)
    stdout_handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processors=[filter_keys, structlog.dev.ConsoleRenderer(colors=config.common.logging_color)],))
    handlers.append(stdout_handler)

    if isinstance(trace_file, str):
        file_handler = logging.FileHandler(trace_file, 'w+')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(processors=[filter_keys,
                                                            structlog.processors.JSONRenderer()],))
        handlers.append(file_handler)

    logging.basicConfig(level=LOG_LEVEL, handlers=handlers)
    logging.getLogger('aiosqlite').setLevel(logging.CRITICAL)
    logging.getLogger('databases').setLevel(logging.CRITICAL)
    logging.getLogger("uvicorn.access").handlers = []
    logging.getLogger("uvicorn.access").propagate = False

    # When logging is fully disabled, also gag uvicorn's own startup loggers,
    # which configure independent handlers and otherwise ignore the root level.
    if LOG_LEVEL > logging.CRITICAL:
        logging.disable(logging.CRITICAL)
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
            uvicorn_logger = logging.getLogger(name)
            uvicorn_logger.handlers = []
            uvicorn_logger.propagate = False
            uvicorn_logger.setLevel(LOG_LEVEL)
