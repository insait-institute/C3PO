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

"""Process management utilities for sandbox execution.

Provides helpers for safe byte decoding, non-blocking async stream reading,
process tree termination, temporary directory management, cgroup memory node
discovery, and child process identification.
"""

import asyncio
import os
from functools import cache

import psutil
import structlog

logger = structlog.stdlib.get_logger()


def try_decode(s: bytes) -> str:
    """Safely decode bytes to a string, returning an error message on failure.

    Args:
        s: The byte sequence to decode.

    Returns:
        The decoded string, or a ``[DecodeError] ...`` message string if
        decoding fails.
    """
    try:
        r = s.decode()
    except Exception as e:
        r = f'[DecodeError] {e}'
    return r


async def get_output_non_blocking(fd):
    """Read available data from an async stream.

    Attempts to read up to 1 MB from the given async file descriptor with a
    5-second timeout.  This is called after the process has already exited or
    been killed, so the read should complete quickly once pipe buffers flush.

    Args:
        fd: An async stream object supporting ``read(n)`` (e.g., an
            ``asyncio.StreamReader``).

    Returns:
        The decoded string content read from the stream, or an empty string
        if nothing was available.
    """
    if fd is None:
        return ''
    res = b''
    try:
        res = await asyncio.wait_for(fd.read(1024 * 1024), timeout=5)
    except (asyncio.TimeoutError, OSError):
        pass
    return try_decode(res)


def kill_process_tree(pid):
    """Kill a process and all of its descendant processes, then reap zombies.

    Uses ``psutil`` to recursively discover all child processes of the given
    PID, sends SIGKILL to each, then waits for them to exit so they don't
    linger as zombies in the process table.

    Args:
        pid: The process ID of the root process to kill.
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return

    procs = children + [parent]
    for p in procs:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # Reap all killed processes so they don't remain as zombies.
    _, alive = psutil.wait_procs(procs, timeout=5)
    for p in alive:
        logger.warning('process still alive after SIGKILL+wait', pid=p.pid)


@cache
def get_tmp_dir() -> str:
    """Return the path to the temporary directory, creating it if needed.

    Reads from the ``SANDBOX_TMP_DIR`` environment variable, falling back
    to ``/tmp``.  In full (Docker) isolation mode, this directory must be
    visible to the Docker host so that bind-mounted sibling containers
    can access the temp files.

    The result is cached so the directory creation and log message only
    occur on the first call.
    """
    TMP_DIR = os.environ.get('SANDBOX_TMP_DIR', '/tmp')
    os.makedirs(TMP_DIR, exist_ok=True)
    logger.info(f'tmp dir: {TMP_DIR}')
    return TMP_DIR


@cache
def get_memory_nodes() -> str:
    """Read the available memory nodes from the cgroup cpuset filesystem.

    Reads ``/sys/fs/cgroup/cpuset/cpuset.mems`` and returns its contents.
    The result is cached after the first call.

    Returns:
        A string representing the available memory node set (e.g., ``'0'``
        or ``'0-3'``).
    """
    with open('/sys/fs/cgroup/cpuset/cpuset.mems') as f:
        return f.read().strip()



def find_child_with_least_pid(ppid) -> int | None:
    """Find the child process with the smallest PID for a given parent PID.

    Iterates over all running processes to find children of ``ppid``, then
    returns the PID of the child with the lowest PID value. This is useful
    for identifying the primary child process spawned by a parent.

    Args:
        ppid: The parent process ID to search children for.

    Returns:
        The PID of the child process with the smallest PID, or ``None`` if
        no children are found or an error occurs.
    """
    try:
        processes = []
        for p in psutil.process_iter(['pid', 'ppid']):
            if p.ppid() == ppid:
                processes.append(p)
        if not processes:
            logger.warning(f'no child process found')
            return None

        child_with_least_pid = min(processes, key=lambda p: p.pid)
        return child_with_least_pid.pid
    except psutil.NoSuchProcess:
        logger.warning(f'no process with PPID {ppid} found.')
        return None
    except Exception as e:
        logger.warning(f'failed to find_child_with_least_pid: {e}')
        return None
