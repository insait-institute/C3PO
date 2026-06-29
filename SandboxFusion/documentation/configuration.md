# Configuration

SandboxFusion is configured via YAML files. The active configuration is selected by the `SANDBOX_CONFIG` environment variable.

## Configuration File Selection

The configuration file path is:

```
sandbox/configs/{SANDBOX_CONFIG}.yaml
```

If `SANDBOX_CONFIG` is not set, it defaults to `"local"`, so the file `sandbox/configs/local.yaml` is loaded.

Examples:

```bash
# Use the default local development config
make run

# Explicitly select a config
SANDBOX_CONFIG=local make run

# Use the docker_full config
SANDBOX_CONFIG=docker_full make run-online
```

## Configuration Schema

The YAML file maps to the `RunConfig` pydantic model defined in `sandbox/configs/run_config.py`. It has three top-level sections:

```yaml
sandbox:
  isolation: lite                  # "lite" or "full"
  max_concurrency: 16              # Max simultaneous sandbox instances (0 = unlimited)
  docker_image: ineil77/sandbox-fusion-server:25042026-2  # Docker image for "full" isolation mode
  default_memory_limit_mb: 8192    # Default memory limit per sandbox execution (MB)
  default_cpu_limit: 2             # Default CPU core limit per sandbox execution

eval:
  max_runner_concurrency: 3  # Max parallel test case evaluations (0 = unlimited)

common:
  logging_color: true        # Enable ANSI color codes in structlog output
```

### sandbox

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `isolation` | `"lite"` or `"full"` | -- (required) | Isolation mode for code execution. See [Isolation Modes](isolation-modes.md). |
| `max_concurrency` | `int` | -- (required) | Maximum number of sandbox instances that may run in parallel. Set to `0` to disable the internal concurrency limiter. |
| `docker_image` | `string` | `"ineil77/sandbox-fusion-server:25042026-2"` | Docker image used when `isolation` is `"full"`. Must have all language runtimes installed. |
| `default_memory_limit_mb` | `int` | `8192` | Default memory limit in megabytes for each sandbox execution. Overridden per-request by `memory_limit_MB` when > 0. |
| `default_cpu_limit` | `float` | `2` | Default CPU core limit for each sandbox execution. In lite mode this sets the CFS quota; in full mode it maps to `docker run --cpus`. |
| `docker_startup_overhead` | `float` | `10` | Extra seconds added to compile/run timeouts in full (Docker) mode to account for container startup latency. Not used in lite mode. |

### eval

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_runner_concurrency` | `int` | `0` | Maximum number of test cases evaluated concurrently when processing a `/submit` request. `0` means no limit. A value like `3` keeps resource usage moderate during development. |

### common

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `logging_color` | `bool` | -- (required) | When `true`, structlog output includes ANSI colour codes for terminal readability. |

## Provided Configuration Files

### local.yaml (default for development)

```yaml
sandbox:
  isolation: lite
  max_concurrency: 16
  docker_image: ineil77/sandbox-fusion-server:25042026-2
  default_memory_limit_mb: 8192
  default_cpu_limit: 2

eval:
  max_runner_concurrency: 3

common:
  logging_color: true
```

- Uses lite isolation (overlayfs + cgroups + network namespaces + chroot).
- Limits to 16 concurrent sandbox instances (reasonable for modern multi-core machines).
- Limits parallel test-case evaluation to 3 to keep resource usage moderate.
- Default resource limits: 8 GB memory and 2 CPU cores per sandbox execution.

### docker_full.yaml (for full/Docker isolation testing)

```yaml
sandbox:
  isolation: full
  max_concurrency: 16
  docker_image: ineil77/sandbox-fusion-server:25042026-2
  default_memory_limit_mb: 8192
  default_cpu_limit: 2

eval:
  max_runner_concurrency: 3

common:
  logging_color: true
```

- Uses full isolation (Docker containers for each execution).
- Selected via `SANDBOX_CONFIG=docker_full` (set automatically by the test harness).
- Limits to 16 concurrent sandbox instances.
- Requires Docker socket and a shared temp directory (`SANDBOX_TMP_DIR`) when running inside a container. See [Isolation Modes](isolation-modes.md) for launch instructions.

### docker_lite.yaml (for lite isolation testing inside Docker)

```yaml
sandbox:
  isolation: lite
  max_concurrency: 16
  docker_image: ineil77/sandbox-fusion-server:25042026-2
  default_memory_limit_mb: 8192
  default_cpu_limit: 2

eval:
  max_runner_concurrency: 3

common:
  logging_color: true
```

- Uses lite isolation (overlayfs + cgroups + network namespaces + chroot) inside a privileged Docker container.
- Selected via `SANDBOX_CONFIG=docker_lite` (set automatically by the test harness).
- Limits to 16 concurrent sandbox instances.

## Creating a Custom Configuration

To create a custom configuration:

1. Create a new YAML file in `sandbox/configs/`, e.g. `sandbox/configs/production.yaml`.
2. Define all three sections (`sandbox`, `eval`, `common`).
3. Set `SANDBOX_CONFIG=production` when starting the server.

Example for a production deployment using Docker-based full isolation:

```yaml
sandbox:
  isolation: full
  max_concurrency: 100
  docker_image: ineil77/sandbox-fusion-server:25042026-2
  default_memory_limit_mb: 8192
  default_cpu_limit: 2

eval:
  max_runner_concurrency: 10

common:
  logging_color: false
```

## Server Startup

The server is started via uvicorn. The `HOST` and `PORT` can be overridden:

```bash
# Default: 0.0.0.0:8080
make run

# Custom host/port
HOST=127.0.0.1 PORT=9090 make run

# Production mode (no hot-reload)
make run-online

# With a specific config
SANDBOX_CONFIG=production make run-online
```

Under the hood, `make run` executes:

```bash
uvicorn sandbox.server.server:app --reload --host 0.0.0.0 --port 8080
```

And `make run-online` executes:

```bash
uvicorn sandbox.server.server:app --host 0.0.0.0 --port 8080
```
