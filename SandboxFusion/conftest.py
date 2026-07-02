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
"""Pytest configuration hooks for the SandboxFusion test suite.

Tests always run against a real server inside a container, mirroring
production.  Two backends are supported:

* ``docker``    — ``docker run`` of the published image.
* ``apptainer`` — ``apptainer run`` of a local ``.sif`` image (useful on
                  HPC systems where Docker is not available).

The ``--sandbox-backend`` option selects the runtime; ``--sandbox-mode``
selects the isolation profile inside the container (``full``, ``lite``,
or ``bindroot``).
"""

import os
import shutil
import signal
import subprocess
import tempfile
import time

import pytest

# Docker state
_container_name: str | None = None

# Apptainer state
_apptainer_proc: subprocess.Popen | None = None
_apptainer_log_path: str | None = None

# Shared state (used by full mode regardless of backend)
_workdir: str | None = None

_DEFAULT_IMAGE_TAG = '25042026-2'
_DEFAULT_DOCKER_IMAGE = f'ineil77/sandbox-fusion-server:{_DEFAULT_IMAGE_TAG}'
_DEFAULT_SIF_NAME = f'sandbox-fusion-server_{_DEFAULT_IMAGE_TAG}.sif'


def pytest_addoption(parser):
    parser.addoption(
        '--sandbox-backend',
        action='store',
        default=None,
        choices=['docker', 'apptainer'],
        help='Container runtime to launch the sandbox server with: '
             '"docker" or "apptainer". Defaults to $SANDBOX_TEST_BACKEND '
             'or "docker".',
    )
    parser.addoption(
        '--sandbox-mode',
        action='store',
        default=None,
        metavar='MODE',
        help='Sandbox isolation profile: "lite", "full", or "bindroot". '
             '"bindroot" is the apptainer-compatible alternative to "lite" '
             'and is the only lite-style mode that works inside apptainer. '
             'Required when launching a server (i.e. when '
             'SANDBOX_TEST_SERVER_URL is unset).',
    )


def _is_xdist_worker(config) -> bool:
    return hasattr(config, 'workerinput')


def pytest_configure(config):
    if _is_xdist_worker(config):
        return

    mode = config.getoption('--sandbox-mode', default=None) \
        or os.environ.get('SANDBOX_TEST_MODE') \
        or os.environ.get('SANDBOX_TEST_DOCKER')  # legacy alias

    backend = config.getoption('--sandbox-backend', default=None) \
        or os.environ.get('SANDBOX_TEST_BACKEND') \
        or 'docker'

    if mode:
        if backend == 'docker':
            _start_docker_server(mode)
        elif backend == 'apptainer':
            _start_apptainer_server(mode)
        else:
            raise pytest.UsageError(f'Unknown --sandbox-backend: {backend!r}')
    elif not os.environ.get('SANDBOX_TEST_SERVER_URL'):
        raise pytest.UsageError(
            'Tests require --sandbox-mode full or --sandbox-mode lite '
            '(optionally with --sandbox-backend docker|apptainer).')


def pytest_unconfigure(config):
    if _is_xdist_worker(config):
        return
    _stop_docker_server()
    _stop_apptainer_server()


def _mode_resources(mode: str) -> tuple[str, str]:
    if mode == 'full':
        return '16g', '8'
    if mode in ('lite', 'bindroot'):
        return '256g', '128'
    raise ValueError(
        f'--sandbox-mode must be "lite", "bindroot", or "full", got {mode!r}')


def _publish_url(url: str, mode: str):
    os.environ['SANDBOX_TEST_SERVER_URL'] = url
    os.environ['SANDBOX_ISOLATION_MODE'] = mode

    from sandbox.tests import client as client_mod
    import httpx
    client_mod.client = httpx.Client(base_url=url, timeout=120)


# ---------------------------------------------------------------------------
# Docker backend
# ---------------------------------------------------------------------------


def _start_docker_server(mode: str):
    global _container_name, _workdir
    import secrets
    _container_name = f'sandbox_test_{secrets.token_hex(4)}'

    image = os.environ.get('SANDBOX_TEST_IMAGE', _DEFAULT_DOCKER_IMAGE)
    port = int(os.environ.get('SANDBOX_TEST_PORT', '18080'))
    container_memory, container_cpus = _mode_resources(mode)

    cmd = [
        'docker', 'run', '-d',
        '--name', _container_name,
        '-p', f'{port}:8080',
        '--memory', container_memory,
        '--cpus', container_cpus,
        '--pids-limit', '4096',
    ]

    if mode == 'full':
        _workdir = tempfile.mkdtemp(prefix='sandbox_work_')
        os.chmod(_workdir, 0o777)
        cmd += [
            '-v', '/var/run/docker.sock:/var/run/docker.sock',
            '-v', f'{_workdir}:{_workdir}',
            '-e', f'SANDBOX_TMP_DIR={_workdir}',
            '-e', 'SANDBOX_CONFIG=docker_full',
        ]
    else:  # lite or bindroot
        sandbox_config = 'docker_bindroot' if mode == 'bindroot' else 'docker_lite'
        cmd += [
            '--privileged',
            '-e', f'SANDBOX_CONFIG={sandbox_config}',
        ]
    cmd.append(image)

    print(f'\n--- Starting sandbox server container ({mode} mode, docker): {_container_name} ---')
    subprocess.run(cmd, check=True, timeout=30)

    url = f'http://localhost:{port}'
    _publish_url(url, mode)
    _wait_for_server(url, timeout=120)
    print(f'--- Server ready at {url} ---\n')


def _stop_docker_server():
    global _container_name, _workdir
    if _container_name is None:
        return
    name = _container_name
    _container_name = None
    print(f'\n--- Stopping sandbox server container: {name} ---')
    try:
        subprocess.run(['docker', 'logs', '--tail', '50', name], timeout=10)
    except Exception:
        pass
    try:
        subprocess.run(
            ['docker', 'rm', '-f', name],
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    if _workdir and os.path.isdir(_workdir):
        shutil.rmtree(_workdir, ignore_errors=True)
        _workdir = None


# ---------------------------------------------------------------------------
# Apptainer backend
# ---------------------------------------------------------------------------


def _resolve_sif_path() -> str:
    sif = os.environ.get('SANDBOX_APPTAINER_SIF')
    if sif:
        return sif
    base = os.environ.get('WORK')
    if not base:
        raise RuntimeError(
            'Apptainer backend requires either SANDBOX_APPTAINER_SIF or '
            '$WORK to be set (image expected at $WORK/' + _DEFAULT_SIF_NAME + ').')
    return os.path.join(base, _DEFAULT_SIF_NAME)


def _start_apptainer_server(mode: str):
    global _apptainer_proc, _apptainer_log_path, _workdir

    sif = _resolve_sif_path()
    if not os.path.isfile(sif):
        raise FileNotFoundError(
            f'Apptainer image not found at {sif!r}. Pull it with:\n'
            f'  apptainer pull docker://ineil77/sandbox-fusion-base:{_DEFAULT_IMAGE_TAG}\n'
            f'  apptainer pull docker://ineil77/sandbox-fusion-server:{_DEFAULT_IMAGE_TAG}')

    # Validate the mode early (same set as docker backend).
    _mode_resources(mode)

    port = int(os.environ.get('SANDBOX_TEST_PORT', '18080'))

    # --cleanenv is essential: without it apptainer inherits the host's
    # environment, and if the launching shell has an activated conda env
    # its compiler toolchain vars (CC/CXX/CPP/LD/AS/AR=x86_64-conda-linux-gnu-*)
    # leak into every sandboxed compile.  go's cgo, D's dmd, and swift's
    # linker then try to invoke x86_64-conda-linux-gnu-cc, which exists only
    # in the host conda env and not in the SIF, so those compiles fail.
    # Docker starts from a clean env and is unaffected; --cleanenv makes
    # apptainer match that.  PORT/SANDBOX_CONFIG are re-injected via --env.
    # --fakeroot is required: the SIF bakes every language toolchain's state
    # into /root (rustup/elan/go/dotnet caches), so the server must appear
    # as uid 0 with HOME=/root; without it 18 language tests fail with
    # "could not create home directory".  --ignore-fakeroot-command keeps
    # that root-mapped user namespace but skips wrapping the container in
    # the `fakeroot` LD_PRELOAD tool, whose single `faked` daemon corrupts
    # its SysV IPC protocol under sustained concurrent process spawning
    # ("libfakeroot internal error: payload not recognized!") and then
    # every new process in the container hangs, wedging the server.
    cmd = ['apptainer', 'run', '--cleanenv', '--fakeroot', '--ignore-fakeroot-command', '--no-home']
    # The server reads PORT from its env (see scripts/run.sh). Apptainer
    # shares the host network namespace, so this binds on the host port.
    cmd += ['--env', f'PORT={port}']

    if mode == 'full':
        _workdir = tempfile.mkdtemp(prefix='sandbox_work_')
        os.chmod(_workdir, 0o777)
        if not os.path.exists('/var/run/docker.sock'):
            raise RuntimeError(
                '--sandbox-mode full requires /var/run/docker.sock on the host '
                '(the sandbox spawns sibling docker containers). On '
                'apptainer-only hosts use --sandbox-mode bindroot.')
        cmd += [
            '-B', '/var/run/docker.sock:/var/run/docker.sock',
            '-B', f'{_workdir}:{_workdir}',
            '--env', f'SANDBOX_TMP_DIR={_workdir}',
            '--env', 'SANDBOX_CONFIG=docker_full',
        ]
    elif mode == 'bindroot':
        # Apptainer-compatible lite-equivalent.  Uses bind mounts in place
        # of overlay, which is the only lite-style mode that works inside
        # apptainer (overlay-on-overlay is rejected by the kernel).
        # Bind the host sandbox/ over the SIF's copy so local edits are
        # picked up without rebuilding the image.
        host_sandbox = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sandbox')
        cmd += [
            '-B', f'{host_sandbox}:/root/sandbox/sandbox',
            '--env', 'SANDBOX_CONFIG=docker_bindroot',
        ]
    else:  # lite
        # Note: lite mode fails inside apptainer (overlay-on-rootfs is
        # rejected with 'failed to clone lowerpath').  Use bindroot
        # instead.  We still expose lite here for parity with docker.
        cmd += ['--env', 'SANDBOX_CONFIG=docker_lite']

    cmd.append(sif)

    _apptainer_log_path = os.path.join(
        tempfile.gettempdir(), f'sandbox_apptainer_{os.getpid()}.log')
    log_file = open(_apptainer_log_path, 'w')

    print(f'\n--- Starting sandbox server ({mode} mode, apptainer): {sif} ---')
    print(f'--- Container log: {_apptainer_log_path} ---')
    _apptainer_proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    url = f'http://localhost:{port}'
    _publish_url(url, mode)
    try:
        _wait_for_server(url, timeout=180, watch_proc=_apptainer_proc,
                         log_path=_apptainer_log_path)
    except Exception:
        _stop_apptainer_server()
        raise
    print(f'--- Server ready at {url} ---\n')


def _stop_apptainer_server():
    global _apptainer_proc, _apptainer_log_path, _workdir
    if _apptainer_proc is None:
        return
    proc = _apptainer_proc
    _apptainer_proc = None
    log_path = _apptainer_log_path
    _apptainer_log_path = None

    print('\n--- Stopping sandbox apptainer server ---')
    if log_path and os.path.isfile(log_path):
        try:
            with open(log_path) as f:
                tail = f.readlines()[-50:]
            for line in tail:
                print(line, end='')
        except Exception:
            pass

    if proc.poll() is None:
        try:
            pgid = os.getpgid(proc.pid)
        except ProcessLookupError:
            pgid = None
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
                proc.wait(timeout=5)
        except ProcessLookupError:
            pass

    if _workdir and os.path.isdir(_workdir):
        shutil.rmtree(_workdir, ignore_errors=True)
        _workdir = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _wait_for_server(url: str, timeout: float,
                     watch_proc: subprocess.Popen | None = None,
                     log_path: str | None = None):
    import httpx
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if watch_proc is not None and watch_proc.poll() is not None:
            tail = ''
            if log_path and os.path.isfile(log_path):
                try:
                    with open(log_path) as f:
                        tail = ''.join(f.readlines()[-40:])
                except Exception:
                    pass
            raise RuntimeError(
                f'Sandbox server process exited early with code '
                f'{watch_proc.returncode}.\n{tail}')
        try:
            r = httpx.get(f'{url}/v1/ping', timeout=5)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f'Sandbox server at {url} did not become healthy within {timeout}s')
