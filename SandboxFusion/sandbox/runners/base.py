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
"""Core execution engine for running code in isolated sandboxes.

This module provides the low-level primitives that every language runner relies
on:

* :func:`run_command_bare` -- execute a single shell command with timeout and
  process-tree cleanup.
* :func:`run_commands` -- orchestrate an optional compile step followed by a
  run step inside either a *lite* (overlayfs + cgroup + netns + chroot) or
  *full* (Docker container) isolation environment.
* :func:`restore_files` -- write base64-encoded file payloads into a working
  directory before execution.
"""

import asyncio
import base64
import os
import shlex
import subprocess
import time
import traceback
from typing import Dict, List, Optional

import psutil
import structlog

from sandbox.configs.run_config import RunConfig
from sandbox.runners.isolation import (CGROUP_VERSION, OBJ_TMPDIR, build_bindroot_wrapper, tmp_bindroot,
                                       tmp_cgroup, tmp_cpuset, tmp_netns, tmp_overlayfs)
from sandbox.runners.types import CodeRunArgs, CodeRunResult, CommandRunResult, CommandRunStatus
from sandbox.utils.execution import get_output_non_blocking, kill_process_tree

logger = structlog.stdlib.get_logger()
config = RunConfig.get_instance_sync()

_cached_base_env: dict | None = None


def _get_base_env() -> dict:
    """Return a cached copy of os.environ to avoid re-copying on every subprocess spawn."""
    global _cached_base_env
    if _cached_base_env is None:
        _cached_base_env = dict(os.environ)
    return _cached_base_env


def _shell_quote(s: str) -> str:
    """Shell-quote a string for safe embedding inside bash -c '...'."""
    return shlex.quote(s)


def _close_subprocess_pipes(p):
    """Close all pipe handles on an asyncio subprocess to release file descriptors."""
    for pipe in (p.stdin, p.stdout, p.stderr):
        if pipe is None:
            continue
        try:
            if hasattr(pipe, 'close'):
                pipe.close()
        except Exception:
            pass


async def run_command_bare(command: str | List[str],
                           timeout: float = 10,
                           stdin: Optional[str] = None,
                           cwd: Optional[str] = None,
                           extra_env: Optional[Dict[str, str]] = None,
                           use_exec: bool = False,
                           preexec_fn=None) -> CommandRunResult:
    """Execute a single command as a subprocess and return its result.

    This is the lowest-level async execution helper.  It creates a subprocess
    (via ``asyncio.create_subprocess_shell`` or ``asyncio.create_subprocess_exec``
    depending on *use_exec*), optionally writes *stdin* to the process, enforces
    *timeout* via ``asyncio.wait_for``, and **always** kills the entire process
    tree after completion (whether the process exited normally, timed out, or
    failed) to prevent orphaned children.  A 5-second grace period is given
    for the asyncio child watcher to reap the top-level process.

    Args:
        command: Shell command string, or argument list when *use_exec* is True.
        timeout: Maximum wall-clock seconds before the process is killed.
        stdin: Optional string to write to the process's standard input.
        cwd: Working directory for the subprocess (shell mode only).
        extra_env: Additional environment variables merged with ``os.environ``.
        use_exec: If True, use ``create_subprocess_exec`` (argument list);
            otherwise use ``create_subprocess_shell`` with ``/bin/bash``.
        preexec_fn: Callable invoked in the child process before exec.

    Returns:
        A :class:`CommandRunResult` with status, timing, exit code, and
        captured stdout/stderr.
    """
    p = None
    try:
        logger.debug(f'running command {command}')
        env = {**_get_base_env(), **(extra_env or {})} if extra_env else _get_base_env()
        if use_exec:
            p = await asyncio.create_subprocess_exec(*command,
                                                     stdin=subprocess.PIPE,
                                                     stdout=subprocess.PIPE,
                                                     stderr=subprocess.PIPE,
                                                     env=env,
                                                     preexec_fn=preexec_fn)
        else:
            p = await asyncio.create_subprocess_shell(command,
                                                      stdin=subprocess.PIPE,
                                                      stdout=subprocess.PIPE,
                                                      stderr=subprocess.PIPE,
                                                      cwd=cwd,
                                                      executable='/bin/bash',
                                                      env=env,
                                                      preexec_fn=preexec_fn)
        start_time = time.time()
        timed_out = False
        if stdin is not None:
            try:
                if p.stdin:
                    p.stdin.write(stdin.encode())
                    # Bound the flush with the run timeout: a child that stays
                    # alive without consuming a stdin larger than the 64 KiB
                    # pipe buffer would otherwise block drain() forever --
                    # before the wall-clock kill below is even armed -- leaking
                    # the execution process and its concurrency slot.
                    await asyncio.wait_for(p.stdin.drain(), timeout=timeout)
                else:
                    logger.warning("Attempted to write to stdin, but stdin is closed.")
            except asyncio.TimeoutError:
                timed_out = True
            except Exception as e:
                logger.exception(f"Failed to write to stdin: {e}")
        if p.stdin:
            try:
                p.stdin.close()
            except Exception as e:
                logger.warning(f"Failed to close stdin: {e}")

        if not timed_out:
            remaining = max(0.0, timeout - (time.time() - start_time))
            try:
                await asyncio.wait_for(p.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                timed_out = True

        execution_time = time.time() - start_time

        if timed_out:
            # Kill the entire process tree and reap zombies in a thread so the
            # synchronous psutil.wait_procs() call doesn't block the event loop.
            await asyncio.to_thread(kill_process_tree, p.pid)
            # Ensure the asyncio child watcher also reaps the top-level process.
            try:
                await asyncio.wait_for(p.wait(), timeout=5)
            except asyncio.TimeoutError:
                logger.warning('process not reaped after kill', pid=p.pid)

        stdout, stderr = await asyncio.gather(
            get_output_non_blocking(p.stdout),
            get_output_non_blocking(p.stderr),
        )
        _close_subprocess_pipes(p)

        if timed_out:
            return CommandRunResult(status=CommandRunStatus.TimeLimitExceeded,
                                    execution_time=execution_time,
                                    stdout=stdout,
                                    stderr=stderr)

        return CommandRunResult(status=CommandRunStatus.Finished,
                                execution_time=execution_time,
                                return_code=p.returncode,
                                stdout=stdout,
                                stderr=stderr)
    except BaseException as e:
        if p is not None:
            kill_process_tree(p.pid)  # sync is OK here; we're already failing
            _close_subprocess_pipes(p)
        if isinstance(e, Exception):
            message = f'exception on running command {command}: {e} | {traceback.print_tb(e.__traceback__)}'
            logger.warning(message)
            return CommandRunResult(status=CommandRunStatus.Error, stderr=message)
        raise


async def run_commands(compile_command: Optional[str], run_command: str, cwd: str, extra_env: Optional[Dict[str, str]],
                       args: CodeRunArgs, **kwargs) -> CodeRunResult:
    """Orchestrate compile and run steps inside an isolated sandbox.

    Depending on the ``RunConfig.sandbox.isolation`` setting this function
    uses one of three isolation strategies:

    * **lite** -- overlayfs (copy-on-write root filesystem), cgroup
      (configurable memory and CPU limits), network namespace, PID
      namespace via ``unshare --pid``, and ``chroot``.
    * **bindroot** -- same as *lite* but with a recursive bind-mount of
      ``/`` (plus per-exec tmpfs scratch) in place of overlay, and
      unprivileged stand-ins for the cgroup/netns layers: rlimits for
      memory/CPU-seconds, ``taskset`` core leases for CPU containment,
      and loopback-only network + PID namespaces via ``unshare``.  Used
      inside apptainer where overlay-on-rootfs is rejected by the kernel.
    * **full** -- ``docker run --rm`` with ``--memory``, ``--cpus``,
      ``--network none``, ``--pids-limit 1024``, and a bind-mount of *cwd*.
      Each container gets a unique ``sandbox_<hex>`` name for reliable
      cleanup.  Commands are shell-quoted via :func:`_shell_quote` and
      wrapped with ``timeout`` inside the container; exit code 124 is
      detected and reported as :attr:`CommandRunStatus.TimeLimitExceeded`.

    Memory and CPU limits default to ``sandbox.default_memory_limit_mb``
    (8192 MB) and ``sandbox.default_cpu_limit`` (2 cores) from the YAML
    config, but can be overridden per-request via ``memory_limit_MB``.

    In full mode, ``sandbox.docker_startup_overhead`` seconds (default 10)
    are added to both compile and run timeouts to account for Docker
    container startup latency, which is not part of the user's code
    execution time.  A ``finally`` block force-removes all containers and
    runs ``chown -R`` to restore file ownership on bind-mounted directories.

    If *compile_command* is provided it is executed first; the run step is
    skipped when compilation fails (non-zero exit or timeout).  After
    execution, any files listed in ``args.fetch_files`` are read from the
    sandbox and returned as base64-encoded strings.

    Args:
        compile_command: Shell command for the compilation step, or ``None``
            to skip compilation.
        run_command: Shell command for the execution step.
        cwd: Working directory inside the sandbox.
        extra_env: Additional environment variables for the subprocess.
        args: A :class:`CodeRunArgs` instance carrying timeouts, stdin, files,
            and the list of files to fetch after execution.
        **kwargs: Forwarded to isolation helpers.  Notable keys include
            ``netns_no_bridge`` (bool), ``disable_pid_isolation`` (bool),
            and ``rlimit_as`` (bool, bindroot only) -- set False for
            runtimes that reserve large virtual address spaces.

    Returns:
        A :class:`CodeRunResult` containing the compile result (if any), the
        run result (if any), and a dict of fetched files.
    """
    files = {}
    compile_res = None
    run_res = None

    if config.sandbox.isolation == 'lite':
        mem_limit = f'{args.memory_limit_MB}m' if args.memory_limit_MB > 0 else f'{config.sandbox.default_memory_limit_mb}m'
        cpu_limit = config.sandbox.default_cpu_limit
        async with tmp_overlayfs() as root, tmp_cgroup(mem_limit=mem_limit, cpu_limit=cpu_limit) as cgroups, tmp_netns(
                kwargs.get('netns_no_bridge', False)) as netns:
            prefix = []
            if CGROUP_VERSION == 2:
                prefix += cgroups
            else:
                for cg in cgroups:
                    prefix += ['cgexec', '-g', cg]
            _needs_sudo = os.getuid() != 0
            if _needs_sudo:
                prefix += ['sudo']
            if not kwargs.get('disable_pid_isolation', False):
                prefix += ['unshare', '--pid', '--fork', '--mount-proc']
            prefix += ['ip', 'netns', 'exec', netns]
            prefix += ['chroot', root]

            def _env_prefix(env: dict | None) -> str:
                """Build an export prefix so env vars survive sudo's env_reset."""
                if not env or not _needs_sudo:
                    return ''
                parts = [f'export {k}={_shell_quote(v)}' for k, v in env.items()]
                return ' && '.join(parts) + ' && '

            if compile_command is not None:
                cmd = f'{_env_prefix(extra_env)}cd {cwd} && {compile_command}'
                compile_res = await run_command_bare(prefix + ['bash', '-c', cmd],
                                                     args.compile_timeout, None, cwd, extra_env, True)
            if compile_res is None or (compile_res.status == CommandRunStatus.Finished and
                                       compile_res.return_code == 0):
                cmd = f'{_env_prefix(extra_env)}cd {cwd} && {run_command}'
                run_res = await run_command_bare(prefix + ['bash', '-c', cmd],
                                                 args.run_timeout, args.stdin, cwd, extra_env, True)

            for filename in args.fetch_files:
                fp = os.path.join(root, os.path.abspath(os.path.join(cwd, filename))[1:])
                if os.path.isfile(fp):
                    with open(fp, 'rb') as f:
                        content = f.read()
                    base64_content = base64.b64encode(content).decode('utf-8')
                    files[filename] = base64_content
            return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)

    elif config.sandbox.isolation == 'bindroot':
        # bindroot is the apptainer-compatible alternative to lite.  On
        # hosts where the launcher lacks /etc/subuid mappings (typical
        # HPC setup), mount syscalls require capabilities the host kernel
        # won't grant us.  We acquire them via nested ``unshare -U -m -r``
        # which gives the wrapper real CAP_SYS_ADMIN in a fresh
        # user+mount namespace.  All bind mounts, the chroot, and the
        # user command run inside that namespace; on exit, the namespace
        # dies and every mount inside it vanishes automatically.
        #
        # cgroup and persistent (bridged) netns setup both require
        # host-level perms that this environment lacks (cgroup tree is
        # root-owned; /run/netns is on an RO filesystem), so unprivileged
        # stand-ins are used instead:
        #
        # * network -- ``unshare -n`` gives a loopback-only netns (the
        #   wrapper brings lo up); no outbound connectivity, no bridge.
        # * PID -- ``unshare -p --fork --mount-proc``; when the ns init
        #   dies every descendant dies with it, so nothing survives
        #   process-tree cleanup.
        # * memory -- RLIMIT_AS sized from ``memory_limit_MB`` (x2 slack
        #   because AS counts virtual memory, which mmap-heavy runtimes
        #   inflate well past their resident usage).  Runners whose VMs
        #   reserve address space far beyond any sane multiplier (tsx's
        #   WASM cages, dotnet's GC regions) opt out via the
        #   ``rlimit_as=False`` kwarg; they keep the RLIMIT_CPU backstop.
        # * CPU -- ``taskset`` pinning onto a leased core group caps how
        #   many cores one exec can occupy, and RLIMIT_CPU backstops the
        #   wall-clock timeout against runaway spinners.
        mem_limit_mb = args.memory_limit_MB if args.memory_limit_MB > 0 else config.sandbox.default_memory_limit_mb
        cores_per_exec = max(1, int(config.sandbox.default_cpu_limit))
        async with tmp_bindroot() as (base_dir, merged), tmp_cpuset(cores_per_exec) as pinned_cores:
            abs_cwd = os.path.abspath(cwd)
            wrapper_script = build_bindroot_wrapper(merged, abs_cwd)

            def _build_cmd(command: str, timeout: float) -> list:
                # Point TMPDIR at the host-backed scratch the wrapper binds at
                # OBJ_TMPDIR so compiler temp objects don't land on the racy
                # tmpfs /tmp (see build_bindroot_wrapper).  OBJ_TMPDIR is not
                # an ancestor of the cwd, so ``go test``'s go.mod handling is
                # unaffected.
                env_exports = f'export TMPDIR={_shell_quote(OBJ_TMPDIR)} && '
                if extra_env:
                    parts = [f'export {k}={_shell_quote(v)}' for k, v in extra_env.items()]
                    env_exports += ' && '.join(parts) + ' && '
                # ulimit -v is in KiB.  -t (CPU seconds) gets 2x the wall
                # timeout + 1 so it only fires on processes that escaped the
                # wall-clock kill, and multi-threaded execs on a 2-core lease
                # aren't penalized.
                rlimits = f'ulimit -t {int(timeout) * 2 + 1}; '
                if kwargs.get('rlimit_as', True):
                    rlimits += f'ulimit -v {int(mem_limit_mb) * 2 * 1024}; '
                inner_cmd = f'{rlimits}{env_exports}cd {abs_cwd} && {command}'
                # ``unshare -U -m -r`` creates a user+mount namespace and
                # maps the caller as uid=0 (real CAP_SYS_ADMIN inside);
                # ``-n`` and ``-p --fork --mount-proc`` add the network and
                # PID namespaces described above.
                # ``bash -c <wrapper> _ <bash> <-c> <inner>`` passes the
                # inner command through to ``chroot ... "$@"`` in the
                # wrapper.
                cmd = [
                    'unshare', '-U', '-m', '-r', '-n', '-p', '--fork', '--mount-proc',
                    'bash', '-c', wrapper_script, '_',
                    'bash', '-c', inner_cmd,
                ]
                if pinned_cores is not None:
                    cmd = ['taskset', '-c', pinned_cores] + cmd
                return cmd

            if compile_command is not None:
                compile_res = await run_command_bare(
                    _build_cmd(compile_command, args.compile_timeout),
                    args.compile_timeout, None, cwd, extra_env, True)
            if compile_res is None or (compile_res.status == CommandRunStatus.Finished and
                                       compile_res.return_code == 0):
                run_res = await run_command_bare(
                    _build_cmd(run_command, args.run_timeout),
                    args.run_timeout, args.stdin, cwd, extra_env, True)

            for filename in args.fetch_files:
                # cwd was rbound into the chroot, so writes inside the
                # chroot reach the host path directly.
                fp = os.path.abspath(os.path.join(cwd, filename))
                if os.path.isfile(fp):
                    with open(fp, 'rb') as f:
                        content = f.read()
                    base64_content = base64.b64encode(content).decode('utf-8')
                    files[filename] = base64_content
            return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)

    elif config.sandbox.isolation == 'full':
        docker_image = config.sandbox.docker_image
        mem_limit = f'{args.memory_limit_MB}m' if args.memory_limit_MB > 0 else f'{config.sandbox.default_memory_limit_mb}m'
        cpu_limit = str(config.sandbox.default_cpu_limit)
        overhead = config.sandbox.docker_startup_overhead
        container_names = []

        def _make_docker_prefix():
            import secrets as _sec
            name = f'sandbox_{_sec.token_hex(8)}'
            container_names.append(name)
            return [
                'docker', 'run', '--rm', '-i',
                '--name', name,
                '--memory', mem_limit,
                '--cpus', cpu_limit,
                '--network', 'none',
                '--pids-limit', '1024',
                '-v', f'{cwd}:{cwd}',
                '-w', cwd,
            ]

        try:
            if compile_command is not None:
                docker_prefix = _make_docker_prefix()
                if extra_env:
                    for k, v in extra_env.items():
                        docker_prefix += ['-e', f'{k}={v}']
                docker_prefix.append(docker_image)
                compile_res = await run_command_bare(
                    docker_prefix + ['bash', '-c', f'timeout {args.compile_timeout} bash -c {_shell_quote(compile_command)}'],
                    args.compile_timeout + overhead, None, cwd, extra_env, True)
                if compile_res and compile_res.status == CommandRunStatus.Finished and compile_res.return_code == 124:
                    compile_res.status = CommandRunStatus.TimeLimitExceeded
                    compile_res.return_code = None
            if compile_res is None or (compile_res.status == CommandRunStatus.Finished and compile_res.return_code == 0):
                docker_prefix = _make_docker_prefix()
                if extra_env:
                    for k, v in extra_env.items():
                        docker_prefix += ['-e', f'{k}={v}']
                docker_prefix.append(docker_image)
                run_res = await run_command_bare(
                    docker_prefix + ['bash', '-c', f'timeout {args.run_timeout} bash -c {_shell_quote(run_command)}'],
                    args.run_timeout + overhead, args.stdin, cwd, extra_env, True)
                if run_res and run_res.status == CommandRunStatus.Finished and run_res.return_code == 124:
                    run_res.status = CommandRunStatus.TimeLimitExceeded
                    run_res.return_code = None
        finally:
            async def _rm_container(name):
                rm = None
                try:
                    rm = await asyncio.create_subprocess_exec(
                        'docker', 'rm', '-f', name,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await asyncio.wait_for(rm.wait(), timeout=10)
                except asyncio.TimeoutError:
                    if rm is not None:
                        rm.kill()
                    logger.warning('docker rm timed out', name=name)
                except Exception:
                    logger.warning('failed to force-remove docker container', name=name)

            if container_names:
                await asyncio.gather(*[_rm_container(n) for n in container_names])
            # Docker containers create files as root inside bind-mounted cwd.
            # Fix ownership so the caller's TemporaryDirectory cleanup succeeds.
            import secrets as _sec
            chown_name = f'sandbox_chown_{_sec.token_hex(8)}'
            fix = None
            try:
                fix = await asyncio.create_subprocess_exec(
                    'docker', 'run', '--rm',
                    '--name', chown_name,
                    '-v', f'{cwd}:{cwd}',
                    docker_image,
                    'chown', '-R', f'{os.getuid()}:{os.getgid()}', cwd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(fix.wait(), timeout=10)
            except asyncio.TimeoutError:
                if fix is not None:
                    fix.kill()
                await _rm_container(chown_name)
                logger.warning('docker chown timed out', cwd=cwd)
            except Exception:
                pass

        for filename in args.fetch_files:
            fp = os.path.abspath(os.path.join(cwd, filename))
            if os.path.isfile(fp):
                with open(fp, 'rb') as f:
                    content = f.read()
                base64_content = base64.b64encode(content).decode('utf-8')
                files[filename] = base64_content
        return CodeRunResult(compile_result=compile_res, run_result=run_res, files=files)


def restore_files(dir: str, files: Dict[str, Optional[str]]):
    """Write base64-encoded file payloads into a directory.

    Each entry in *files* maps a relative path to its base64-encoded content.
    Intermediate directories are created as needed.  Entries are silently
    skipped if the content is not a string (e.g. ``None``) or the filename
    contains the sentinel ``IGNORE_THIS_FILE``.

    Args:
        dir: Target directory to write files into.
        files: Mapping of ``{relative_path: base64_content}``.
    """
    for filename, content in files.items():
        if not isinstance(content, str):
            continue
        if "IGNORE_THIS_FILE" in filename:
            continue
        filepath = os.path.join(dir, filename)
        dirpath = os.path.dirname(filepath)
        os.makedirs(dirpath, exist_ok=True)
        with open(filepath, 'wb') as file:
            file.write(base64.b64decode(content))
