# Isolation Modes

SandboxFusion provides two isolation modes to control how user-submitted code is executed. The mode is selected via the `sandbox.isolation` field in the YAML configuration file.

## Overview

| Mode | Mechanism | Overhead | Platform | Privileges |
|------|-----------|----------|----------|------------|
| `lite` | overlayfs + cgroups v1/v2 + network namespaces + PID namespace + chroot | ~100ms | Linux only | Requires root (sudo) or privileged container |
| `full` | Docker containers with memory, CPU, network, and PID limits | ~500ms+ | Any platform with Docker | Requires Docker daemon access |

Both modes execute untrusted code in an isolated environment. Neither mode uses nested Docker.

---

## Lite Isolation

Lite isolation is the default mode. It provides fast, lightweight isolation using Linux kernel primitives directly (no container runtime involved).

### What It Does

When a code execution request arrives, the sandbox:

1. **Creates a temporary directory** (under the path set by `SANDBOX_TMP_DIR`, defaulting to `/tmp`) for the code and any uploaded files.
2. **Sets up an overlayfs mount**: The host filesystem is mounted read-only as the lower layer, with a disposable upper layer for writes. The executed code sees a copy-on-write view of the filesystem -- any modifications (including to system files) are discarded when execution completes.
3. **Creates a cgroup** (v1 or v2, auto-detected): Applies memory and CPU limits to the sandbox process. Memory defaults to `sandbox.default_memory_limit_mb` (8192 MB), overridden per-request by `memory_limit_MB`. CPU defaults to `sandbox.default_cpu_limit` (2 cores).
4. **Enters a network namespace**: The sandbox process is placed in a separate network namespace with NAT-bridged outbound connectivity (or no connectivity if `--no-bridge` is set).
5. **Enters a PID namespace**: Uses `unshare` to give the sandbox process its own PID namespace, preventing it from seeing or signaling host processes.
6. **Chroots** into the overlayfs mount: The sandbox process sees the overlayfs as its root filesystem, isolating it from the real filesystem.
7. **Executes the code** within this isolated environment with the configured timeouts.
8. **Tears down** the overlayfs (each unmount isolated so one failure cannot cascade, plus a `/proc/mounts` safety sweep), cgroup (killing all surviving processes and reaping zombies), and namespace (always returning the subnet to the pool) after execution completes (or on timeout).

### Requirements

- **Linux**: overlayfs, cgroups v1/v2 (auto-detected), and namespaces are Linux kernel features.
- **Root / privileged**: Setting up overlayfs and cgroups requires root privileges. When running inside Docker, the container must be started with `--privileged`.
- **Network namespace scripts**: The scripts `scripts/create_net_namespace.sh` and `scripts/clean_net_namespace.sh` manage the network namespace lifecycle. Veth interface names are truncated (e.g., `ve-a1b2c3` / `ve-a1b2c3-br`) to comply with Linux's 15-character interface name limit.

### Configuration

```yaml
sandbox:
  isolation: lite
  max_concurrency: 16
```

### Implementation

The isolation primitives are implemented in `sandbox/runners/isolation.py`:

- `tmp_overlayfs()` -- Async context manager that creates an overlayfs mount with the host root (`/`) as the read-only lower layer and a tmpfs upper layer. Mounts `/proc`, `/sys`, and `/dev` inside the overlay. On teardown, each unmount is isolated in its own try/except, followed by a safety-net sweep of `/proc/mounts`. If setup fails partway through, all partial mounts are cleaned up before the exception propagates.
- `tmp_cgroup(mem_limit, cpu_limit)` -- Async context manager that creates a cgroup (v1 or v2, auto-detected) with the specified limits. Cleanup is in a `finally` block that kills all contained processes and removes the cgroup. Cgroup v2 initialization is protected by a `threading.Lock` for thread safety.
- `tmp_netns(no_bridge)` -- Async context manager that allocates a subnet from a thread-safe pool, creates a network namespace, and always returns the subnet in a `finally` block even if teardown fails. Raises `RuntimeError` after 60 seconds if no subnet is available.

These are composed together in `sandbox/runners/base.py:run_commands()` when the isolation mode is `lite`.

---

## Full Isolation (Docker)

Full isolation runs each code execution inside a disposable Docker container. This provides the strongest isolation and works on any platform with Docker.

### What It Does

When a code execution request arrives, the sandbox:

1. **Launches a Docker container** from the configured `docker_image` (default: `ineil77/sandbox-fusion-server:25042026-2`) with these flags:
   - `--rm` -- Container is automatically removed after exit.
   - `--name sandbox_<random_hex>` -- Unique container name for reliable force-removal on cleanup.
   - `--memory` -- Memory limit from the request or `sandbox.default_memory_limit_mb` (default 8192 MB).
   - `--cpus` -- CPU limit from `sandbox.default_cpu_limit` (default 2 cores).
   - `--network none` -- Complete network isolation (no egress traffic possible).
   - `--pids-limit 1024` -- Limits the number of processes/threads to prevent fork bombs. Set to 1024 to accommodate toolchains like Lean's LLVM linker that spawn many threads.
2. **Mounts the temporary directory** containing the code and uploaded files into the container via `-v <workdir>:<workdir>`.
3. **Adds `docker_startup_overhead`** seconds (default 10) to timeouts to account for Docker startup latency separate from code execution time. The user's code is wrapped with the `timeout` command inside the container; exit code 124 is detected and reported as `TimeLimitExceeded`.
4. **Executes the compile and run commands** inside the container (each in a separate container with its own unique name). All commands are shell-quoted via `shlex.quote()` to prevent injection.
5. **Collects results** (stdout, stderr, exit code, fetched files) from the bind-mounted host directory.
6. **Force-removes all containers** via `docker rm -f` in a `finally` block, then runs `chown -R` to restore file ownership on bind-mounted directories, ensuring no leaked containers or permission issues even on unexpected errors.

### Requirements

- **Docker daemon**: Must be installed and running on the host.
- **`ineil77/sandbox-fusion-server:25042026-2` image**: The Docker image must have all language runtimes pre-installed. Build it with `make build-server-image`.
- **No nested Docker**: Full isolation does NOT use Docker-in-Docker. The host Docker daemon creates sibling containers (Docker-out-of-Docker).
- **Shared temp directory**: When the server runs inside Docker, mount a shared temp directory (e.g., `-v /tmp/sandbox-shared:/tmp/sandbox-shared`) and set `-e SANDBOX_TMP_DIR=/tmp/sandbox-shared` so that sibling execution containers can access the temp directories created by the server. The bind mount source is resolved on the Docker **host**, so the path must exist on the host and be shared between the server container and execution containers. Without this, all executions fail with "No such file or directory".
- **Docker socket**: The server container needs `-v /var/run/docker.sock:/var/run/docker.sock` to communicate with the host Docker daemon.

### Configuration

```yaml
sandbox:
  isolation: full
  max_concurrency: 100
  docker_image: ineil77/sandbox-fusion-server:25042026-2
  default_memory_limit_mb: 8192
  default_cpu_limit: 2
```

The `docker_image` field specifies which image to use for execution containers. It defaults to `ineil77/sandbox-fusion-server:25042026-2`, which is built by `make build-server-image` and includes all 20+ language runtimes.

**Launching for full mode (Docker-out-of-Docker):**

```bash
docker run -d --rm --privileged \
    -p 8080:8080 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /tmp/sandbox-shared:/tmp/sandbox-shared \
    -e SANDBOX_CONFIG=full_test \
    -e SANDBOX_TMP_DIR=/tmp/sandbox-shared \
    ineil77/sandbox-fusion-server:25042026-2
```

---

## Choosing a Mode

| Consideration | Lite | Full |
|---------------|------|------|
| **Performance** | Fast (~100ms overhead) | Slower (~500ms+ overhead per execution) |
| **Isolation strength** | Strong (kernel-level) but shares kernel with host | Strongest (full container boundary) |
| **Platform** | Linux only | Any OS with Docker |
| **Development** | Good for local Linux dev, CI | Good for any platform |
| **Production** | Recommended when running inside Docker already | Recommended for bare-metal hosts |

### Typical deployment patterns

**Docker deployment (recommended):**
- Build `ineil77/sandbox-fusion-server:25042026-2` which runs the SandboxFusion server inside a Docker container.
- The server uses **lite** isolation inside the container (overlayfs + cgroups within the privileged container).
- The container needs `--privileged` for overlayfs/cgroups.
- No nested Docker involved.

**Bare-metal Linux deployment:**
- Install runtimes directly on the host (`make install-runtimes`).
- Use **lite** isolation for best performance.
- Requires root/sudo for overlayfs and cgroups.

**Full isolation deployment:**
- The SandboxFusion server runs on the host (or in a container with Docker socket access and a shared temp directory via `-v /tmp/sandbox-shared:/tmp/sandbox-shared -e SANDBOX_TMP_DIR=/tmp/sandbox-shared`).
- Each code execution spawns a separate `ineil77/sandbox-fusion-server:25042026-2` container via the host Docker daemon.
- Use when you want maximum isolation and can tolerate higher latency.

### Behavioral differences between modes

| Behavior | Lite | Full |
|----------|------|------|
| **Network** | NAT-bridged outbound connectivity | Fully isolated (`--network none`) -- no egress traffic |
| **File retrieval** | overlayfs captures all filesystem writes (absolute and relative paths) | Only files in the bind-mounted working directory survive; writes to absolute paths outside the workdir are lost |
| **PID limit** | No hard limit (PID namespace via `unshare`) | `--pids-limit 1024` |
| **Resource pooling** | No pooling -- fresh overlayfs, cgroup, and netns created per execution (~100ms), torn down after | No pooling -- full Docker lifecycle on every execution |

---

## Execution Flow

Regardless of isolation mode, the general code execution flow is:

1. Create a temporary directory (under `SANDBOX_TMP_DIR`, defaulting to `/tmp`).
2. Write files passed through the `files` parameter.
3. Write the `code` parameter to a temporary file (e.g. `/tmp/tmpha4dcl5b/tmpx8k1pnfh.py`).
4. Set up the isolation environment (overlayfs + cgroup + namespace for lite, or Docker container for full).
5. Execute compilation commands (if the language requires compilation).
6. Execute the run command.
7. Retrieve files specified by `fetch_files`.
8. Tear down the isolation environment and clean up the temporary directory.

File paths in `files` and `fetch_files` support both absolute and relative paths. Relative paths are resolved relative to the temporary directory.

**Note on `fetch_files` in full mode:** In full mode, only files within the bind-mounted working directory (the temp dir under `/tmp`) can be retrieved. Files written to absolute paths outside this directory (e.g., `/mnt/output`) exist only inside the ephemeral container and are lost when it exits. In lite mode, overlayfs captures all writes regardless of path, so absolute-path files can be retrieved.
