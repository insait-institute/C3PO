# Apptainer migration

SandboxFusion was originally built to run as a Docker container, using Docker
both to **host the server** and (in `full` isolation mode) to **spawn a fresh
sibling container per code execution**. On HPC systems Docker is typically
unavailable: there is no Docker daemon, users are unprivileged, and the only
supported container runtime is [Apptainer](https://apptainer.org/) (formerly
Singularity) running rootless via `--fakeroot`.

This document describes the changes made to run SandboxFusion under Apptainer:
the new **`bindroot` isolation mode** that replaces overlay-based `lite` mode,
why it was necessary, what it does, and the fixes required to make the full
test suite pass under Apptainer at parity with Docker.

---

## 1. Background: the three isolation modes

`RunConfig.sandbox.isolation` (`sandbox/configs/run_config.py`) now accepts
three values:

| Mode | Mechanism | Per-exec isolation | Runtime requirement |
|------|-----------|--------------------|---------------------|
| `full` | `docker run --rm` of a fresh container per execution | container (memory/cpu cgroups, `--network none`, `--pids-limit`) | Docker daemon + `/var/run/docker.sock` |
| `lite` | overlayfs (host `/` as lower, tmpfs upper) + `chroot` + cgroup + netns + `unshare --pid` | overlay CoW root, cgroup limits, network namespace | privileged container, overlay-on-rootfs allowed |
| `bindroot` | recursive bind-mount of `/` + per-exec tmpfs scratch + `chroot`, all inside a nested `unshare -U -m -r` | unshared mount + user namespace, unique cwd | **Apptainer-compatible** — no Docker, no host root |

`full` and `lite` are the original Docker modes. `bindroot` is the new mode for
Apptainer. The test harness selects them via `--sandbox-backend
{docker,apptainer}` and `--sandbox-mode {full,lite,bindroot}`
(`conftest.py`); the matching server config files are `docker_full.yaml`,
`docker_lite.yaml`, and the new `docker_bindroot.yaml`.

---

## 2. Why `bindroot` was necessary

`lite` mode builds the sandbox filesystem with **overlayfs**, using the host
root `/` as the read-only lower layer and a tmpfs upper layer so all writes are
ephemeral:

```
mount -t overlay overlay -o lowerdir=/,upperdir=...,workdir=... merged/
```

This **fails inside Apptainer**. Apptainer's own `/` is *already* an overlay
(the SIF's squashfs with a session tmpfs on top) whose child mounts are locked
(`/etc/hosts`, `/etc/resolv.conf`, `/.singularity.d/*`, `/dev/*`). The kernel
refuses to stack a second overlay whose lower layer is such a tree:

```
overlayfs: failed to clone lowerpath
```

So a different filesystem-isolation strategy was required for Apptainer.
`bindroot` provides the same *lite-equivalent* semantics (ephemeral, isolated
root; fast per-exec setup) without overlayfs.

A second Apptainer constraint shaped the design: on HPC hosts the user is
usually **not listed in `/etc/subuid`**, so Apptainer's `--fakeroot` falls back
to an `LD_PRELOAD`-based fakeroot rather than a real uid mapping. Under that
fallback the kernel sees the *real* uid for `mount(2)` syscalls and rejects them
with `EPERM`, regardless of fakeroot's `uid=0` illusion. The workaround is to
do the mounts inside a **nested `unshare -U -m -r`**, which gives the process
real `CAP_SYS_ADMIN` in a fresh user + mount namespace.

---

## 3. What `bindroot` does

Implementation lives in `sandbox/runners/isolation.py`
(`build_bindroot_wrapper`, `tmp_bindroot`) and `sandbox/runners/base.py`
(`run_commands`, the `isolation == 'bindroot'` branch).

For each code execution the server:

1. Allocates an empty scratch base dir on the host
   (`/tmp/bindroot_<rand>/merged`) via `tmp_bindroot()`.
2. Builds a bash "wrapper" snippet and runs the whole thing as:

   ```
   unshare -U -m -r bash -c '<wrapper>' _ bash -c '<inner cmd>'
   ```

   `unshare -U -m -r` creates a fresh **user + mount namespace** with the caller
   mapped to `uid=0` (real `CAP_SYS_ADMIN` inside), so the mounts below are
   permitted even without `/etc/subuid` entries.

3. The wrapper (`build_bindroot_wrapper`) sets up the chroot:

   ```
   mount --make-rprivate /                 # detach from host shared propagation
   mount --rbind / merged/                 # recursive bind of '/' as the new root
   mount --make-rprivate merged/
   mount -t tmpfs tmpfs merged/tmp         # ephemeral, isolated /tmp
   mount -t tmpfs tmpfs merged/var/tmp     # (+ /run, /root/.cache)
   ...
   mount --rbind <cwd> merged/<cwd>        # re-expose the host cwd through the tmpfs
   exec chroot merged "$@"                 # run the user command
   ```

   Because all of this lives in the unshared mount namespace, the entire mount
   tree **vanishes automatically when the namespace dies** — there is nothing to
   unmount on the happy path. `tmp_bindroot`'s cleanup only sweeps stray mounts
   left by an aborted setup, then `rmtree`s the base dir.

4. `restore_files` stages input files into the host cwd before exec; because the
   cwd is `--rbind`'d into the chroot at the same absolute path, writes made
   inside the chroot land directly on the host path, so `fetch_files` works
   without copying back out.

### Resource and network isolation: unprivileged stand-ins for cgroups/netns

`lite` mode's resource isolation relies on primitives an unprivileged Apptainer
user does not have: the cgroup tree is root-owned and not delegated
(`tmp_cgroup` impossible), and `/run/netns` sits on a read-only filesystem
(`tmp_netns` impossible). The initial migration simply skipped both — sandboxed
code ran with unlimited memory/CPU and full network access, and the resulting
free-for-all contention caused several of the test flakes in §5. A later
hardening pass replaced each skipped layer with an unprivileged equivalent:

- **Memory:** `ulimit -v` sized from `memory_limit_MB` (×2 slack for virtual
  address space). Runtimes that reserve VAS far beyond any sane multiplier opt
  out via `run_commands(rlimit_as=False)` — tsx's WASM cages (typescript) and
  the .NET GC (csharp) — keeping only the `RLIMIT_CPU` backstop.
- **CPU:** `taskset` pinning onto per-exec core groups leased from the affinity
  mask (`tmp_cpuset`), plus `ulimit -t` as a runaway-spinner backstop.
  `max_concurrency` is sized to cores / `default_cpu_limit` (32 on a 64-core
  allocation at 2 cores/exec — see `docker_bindroot.yaml`).
- **Network:** `unshare -n` gives each exec a loopback-only namespace; the
  wrapper brings `lo` up so localhost servers still work. Egress is blocked by
  design (the egress test skips under bindroot; a companion test asserts the
  block actually holds).
- **PID:** `unshare -p --fork --mount-proc`, so no process survives the death
  of the namespace's init.

---

## 4. Launch / harness changes

- **`Makefile`**: new `pull-apptainer-images`, `start-apptainer-container`,
  `test-apptainer-bindroot`, and `test-rl-load` targets. The standalone launcher
  binds the host `sandbox/` over the SIF's copy
  (`-B $(CURDIR)/sandbox:/root/sandbox/sandbox`) so local code edits are picked
  up without rebuilding the SIF. It runs with
  `--cleanenv --fakeroot --ignore-fakeroot-command --no-home` (see §6.1 for why
  that exact flag combination) and `SANDBOX_LOG_LEVEL=OFF` (server log level is
  configurable via that env var; default `INFO`).
- **`conftest.py`**: `--sandbox-backend {docker,apptainer}` and `--sandbox-mode
  {full,lite,bindroot}` options; an Apptainer server launcher
  (`_start_apptainer_server`) that resolves the SIF from `$SANDBOX_APPTAINER_SIF`
  or `$WORK`, starts it with `apptainer run`, and waits on `/v1/ping`.
- **`docker_bindroot.yaml`**: server config selecting `isolation: bindroot`.

> **Testing gotcha:** killing the `apptainer run` launcher pid does **not** kill
> the `uvicorn` server child — a stale server keeps binding the port and
> silently answers `/v1/ping`, so a "restart" can quietly keep serving old code.
> Kill by listener instead: `ss -ltnp | grep :<port>` → kill that uvicorn pid.

---

## 5. Fixes required to pass the test suite under Apptainer

`make test-apptainer-bindroot` initially had 11 failures that did not occur
under Docker. Four independent root causes; all now green (213 passed,
2 skipped).

### 5.1 Host conda environment leaked into the SIF — `--cleanenv`

Apptainer **inherits the launching shell's environment by default**. When the
launching shell has an active conda environment, its compiler-toolchain
variables leak into the container and then into every sandboxed compile:

```
CC = CXX = CPP = LD = AS = AR = x86_64-conda-linux-gnu-*
CONDA_PREFIX = /…/micromamba/envs/<env>
```

Go's cgo, D's `dmd`, and Swift's linker read `CC`/`CXX`/`LD` and try to invoke
`x86_64-conda-linux-gnu-cc`, which exists only in the host conda env, **not in
the SIF** → compile failures (`cgo: C compiler … not found`, `linker exited with
status …`). Docker starts from a clean environment and is unaffected.

**Fix:** add `--cleanenv` to the Apptainer invocations (`conftest.py`
`_start_apptainer_server` and the Makefile `start-apptainer-container`). `PORT`
and `SANDBOX_CONFIG` are re-injected explicitly via `--env`. This makes
Apptainer match Docker's clean-env behaviour and fixed ~6 tests (go_test,
D_ut, swift).

### 5.2 cpp/swift `.o` corruption under concurrency — host-backed `TMPDIR`

Under heavy mixed-language load, ~4% of C++/Swift compiles failed with:

```
ld: cannot find /tmp/ccXXXX.o: file format not recognized
collect2: error: ld returned 1 exit status
```

The assembler writes its intermediate object to the chroot's **tmpfs `/tmp`**,
and the linker then reads it back empty/corrupt. This is *not* OOM (the box has
512 GB and there were no OOM kills) — the chroot's per-exec tmpfs `/tmp`
intermittently loses files written to it mid-build when many bindroot sandboxes
churn mounts concurrently. Docker is immune because each execution is a separate
container with its own stable `/tmp`. It does **not** reproduce with standalone
`unshare` loops — only the running server under mixed-language load triggers it.

**Fix:** give compilers a **host-backed (xfs) `TMPDIR`** instead of the racy
tmpfs:

- `sandbox/runners/isolation.py`: new constant `OBJ_TMPDIR = '/tmp/.sandbox_obj'`;
  `build_bindroot_wrapper` bind-mounts a per-exec host directory
  (`<base_dir>/tmproot`) onto `OBJ_TMPDIR` inside the chroot. `/tmp` itself stays
  a tmpfs (see §5.4 for why).
- `sandbox/runners/base.py`: the bindroot `_build_cmd` exports
  `TMPDIR=$OBJ_TMPDIR` for the compile and run commands.

Verified at 160/160 cpp and 40/40 swift under heavy concurrent mixed load.

### 5.3 Test timeouts too tight for bindroot

With no cgroup CPU isolation (§3), compiles contend for CPU under the 16-worker
test load, so a few timeout-oriented tests that assume a fast compile became
flaky:

- `test_lean_error` used the **default 10 s** run timeout while its sibling
  `test_lean_pass` used 30 s for the same Mathlib import → bumped to 30 s.
- `test_rust_timeout` used `compile_timeout=1` (rustc can't reliably finish in
  1 s under load) → bumped to 20 s.

Both tests verify **run-timeout** behaviour; the compile budget was incidentally
too tight. (`sandbox/tests/runners/test_lean.py`, `test_rust.py`.)

### 5.4 Constraint discovered while fixing 5.2 — `go test` and `os.TempDir()`

`go test` **ignores a `go.mod` when its module root equals `os.TempDir()`**
(Go source `cmd/go/internal/modload/init.go`:
`search.InDir(modRoot, os.TempDir()) == "."`). The bindroot working directory is
under `/tmp`, so:

- Pointing `TMPDIR` at the cwd makes `os.TempDir() == modRoot` → go ignores the
  staged `go.mod` → `no required module provides package …`.
- Making `/tmp` *itself* a host bind (instead of tmpfs) also breaks `go test` —
  the tmpfs↔xfs device boundary at `/tmp` is what keeps go from treating the cwd
  as a throwaway temp module.

This is why the §5.2 fix **keeps `/tmp` a tmpfs** and points `TMPDIR` at a
*separate* host-backed directory (`/tmp/.sandbox_obj`) that is **not** the cwd.
With that arrangement, cpp/swift get stable compiler temps **and** `go test`
keeps working (40/40 under load).

### Defensive: `mount --make-rprivate /`

The wrapper makes the namespace's mount tree private before laying down per-exec
mounts. On Apptainer this is a no-op (its `/tmp` is already private), but it is
correct hygiene for hosts where `/` is `rshared` (e.g. systemd defaults under a
privileged docker `bindroot` run), preventing a concurrent exec's mount from
propagating in and shadowing this sandbox's `/tmp` mid-build.

---

## 6. Production hardening under sustained RL load

The test suite passes in minutes; an RL training run hammers the server for
hours with ~48 concurrent streams of adversarial, model-generated code. Two
server-wedging bugs only surfaced under that regime — both manifested as the
training client's read timeouts ramping up over a few RL steps until every
reward was zero, while the server still answered `/v1/ping` (and even trivial
`/run_code` probes) normally.

### 6.1 `faked` daemon corruption — `--ignore-fakeroot-command`

`--fakeroot` is required: the SIF bakes every language toolchain's state into
`/root` (rustup/elan/go/dotnet caches), so the server must appear as uid 0 with
`HOME=/root` — without it, 18 language tests fail with "could not create home
directory". But on hosts without `/etc/subuid` entries, `--fakeroot` *also*
wraps the container in the `fakeroot` LD_PRELOAD tool, whose single `faked`
daemon serializes all metadata faking over one SysV IPC channel. Under
sustained concurrent process spawning the daemon corrupts its protocol
(`libfakeroot internal error: payload not recognized!`), after which **every
new process in the container hangs** — the whole server wedges.

**Fix:** add `--ignore-fakeroot-command` alongside `--fakeroot`. This keeps the
root-mapped user namespace (uid 0, `HOME=/root`) but skips the LD_PRELOAD
wrapper entirely. The flag is meaningless without `--fakeroot`; the two must be
used together. Applied in the Makefile launcher and the `conftest.py` test
fixture.

### 6.2 stdin-drain concurrency-slot leak

In `run_command_bare` (`sandbox/runners/base.py`), stdin was written to the
child with an unbounded `await p.stdin.drain()` *before* the wall-clock
`wait_for(p.wait(), timeout)` was armed. A child that stays alive without
consuming a stdin larger than the 64 KiB pipe buffer blocks `drain()` forever:
the handler never reaches the kill, never returns, and **permanently leaks its
`max_concurrency` semaphore slot**. Each occurrence shrinks effective capacity
by one; the process burns no CPU, so `ulimit -t` never fires, and trivial
health probes keep succeeding through the remaining free slots — which is why
liveness watchdogs stayed silent while training rewards collapsed.

**Fix:** bound the stdin flush with the run timeout and charge its elapsed time
against the subsequent `p.wait()` window, so a poisoned stdin yields a normal
`TimeLimitExceeded` instead of a leaked slot.

### 6.3 RL-load regression suite — `make test-rl-load`

`sandbox/tests/test_rl_load.py` (marker: `stress`, excluded from the default
run) replays the training access pattern against a live server so these
regressions are caught without submitting GPU jobs:

- **storm**: 288 requests over 48 concurrent client streams of mixed
  workloads (fast, CPU-heavy, TLE, compile-error, big-output, poison-stdin)
  with a background liveness prober; zero client timeouts allowed.
- **poison-stdin slot leak**: 12 concurrent sleepers fed 1 MiB of stdin must
  all return `TimeLimitExceeded` (the pre-fix server fails this in seconds).
- **kill churn**: 3 waves × 32 infinite loops, exercising timeout-kill paths
  at full concurrency.

---

### 6.4 Read-timeout storms under sustained load — admission control + retry redesign

A long `codegenrl` run (`~/SlurmLogs/codegenrl.out`) surfaced a *softer* failure
than the §6.1/§6.2 wedges: the client logged **753 "Read timed out (read
timeout=30)" warnings** over ~10 hours. Crucially this was **not** a recurrence
of the wedge:

- `critic/score/mean` stayed healthy (0.15–0.42) across all 84 steps — rewards
  never collapsed to zero.
- Only **1** of the 753 timeouts exhausted all retries; the rest recovered.
- The timeouts came in **steady bursts** (8–10/min spikes at reward time, then
  quiet), not a monotonic ramp.
- The run ultimately died from an unrelated **NCCL collective-timeout** (a 600 s
  `_ALLGATHER_BASE` watchdog on one rank — the checkpoint-save desync), not from
  the sandbox.

**Root cause of the timeouts: queue saturation, not a server fault.** The
trainer runs 8 `RewardLoopWorker`s each holding `max_concurrent` in-flight
requests (32 → **256 in-flight**) against a server with **48 execution slots**
(`nproc/2` on the 96-core standing allocation). The ~208 excess requests queue
on the server's `max_concurrency` semaphore; under a reward-time burst the tail
waits longer than the client's 30 s read timeout (`compile_timeout + run_timeout
+ API_TIMEOUT` = 10+10+10).

Two facts about this regime are worth pinning down because they are
counter-intuitive:

- **Raising `max_concurrency` does not help.** At `default_cpu_limit=2`, 48 slots
  already pin all 96 cores; more slots just oversubscribe CPU so each exec runs
  slower. `bench_concurrency.py` confirms throughput plateaus at ~34 req/s from
  256 in-flight onward — the server is **CPU-throughput-bound**, not slot-bound.
- **The old benchmark under-predicted real load (now rewritten).** The same
  48-slot / 256-in-flight point showed *zero* timeouts in the old bench but 753
  in training, because it replayed independent toy snippets (one request per
  "sample", levels capped at 96) and *drained between levels*, whereas a real
  Code-Contests batch submits a continuous burst of *many test cases per sample*
  with far more slow/TLE solutions early in training. `bench_concurrency.py` has
  been rewritten to be a one-to-one reflection (see §6.5).

The danger was not the timeouts themselves (retries absorbed them) but the
**retry schedule**: the verl client treated a read timeout as "server wedged /
restarting (minutes)" and backed off **30/60/90/120 s** (up to ~7 min for the
case that hit 5/5). Reward aggregation waits for *all* cases, so one saturated
case stalls the whole reward step for minutes — the most plausible path from a
transient timeout to the 600 s NCCL watchdog.

**Fix — server-side admission control.** `RunConfig.sandbox.queue_timeout`
(env `SANDBOX_QUEUE_TIMEOUT`, default `0` = off) bounds how long `/run_code`
waits for a slot. `_acquire_slot` (`sandbox/server/sandbox_api.py`) does
`asyncio.wait_for(sem.acquire(), queue_timeout)`; on expiry it returns **HTTP
503** instead of holding the connection. Set it below the client read timeout
minus the run/compile budget (15 s for verl's 30 s read timeout). An
over-subscribed server now sheds load fast and retryably rather than letting the
client read-time-out. The standing-server launcher
(`~/BashScripts/sandbox_server.sh`) sets `SANDBOX_QUEUE_TIMEOUT=15`.

**Fix — client retry redesign** (`verl/.../sandbox_fusion/utils.py`). Two
transient modes are now distinguished, both with **capped exponential backoff +
full jitter** (jitter desynchronises the many cases that time out on the same
tick, preventing a synchronised re-burst):

- *overload* — HTTP **503** (admission-rejected) or 504: server healthy but
  busy, queue drains in seconds → fast backoff (base 1 s, cap 8 s).
- *unreachable* — read timeout / connection error: now rare, usually a real
  restart → moderate backoff (base 2 s, cap 30 s).

The total budget is kept well under the 600 s NCCL watchdog so a stuck case can
never stall a reward step into a collective-timeout crash. All knobs are
env-overridable (`SANDBOX_FUSION_*`).

**Fix — raise the NCCL collective watchdog for margin.** The crash that ended
the run was exactly this failure mode: one rank stalled in the reward path
(under the *old* 30/60/90/120 s backoff a single case that hit all 5 retries ran
~540 s + overhead) past the FSDP process group's 600 s collective watchdog, so
the other ranks aborted the whole job at the next `_ALLGATHER_BASE` with a
spurious timeout — at step 85, ~20 steps *after* `global_step_64` had saved
cleanly (it was never a checkpoint-save bug). The bounded backoff above now
keeps the reward worst-case well under 600 s, but the recipe
(`verl/recipes/run_cco.sh`) also sets `actor_rollout_ref.nccl_timeout=1800`
(env `NCCL_TIMEOUT`) so a legitimately slow reward — e.g. riding out a real
sandbox restart — degrades into a slow step rather than a crash.

> **Sizing note.** The real lever for fewer timeouts is *throughput*: more
> sandbox cores (a bigger / second standing allocation) or cheaper rewards
> (lower `run_timeout`, fewer test cases), not a higher `max_concurrency` on the
> same node.

### 6.5 Reward verdict, short-circuit, and a faithful benchmark

Two changes on the verl side both improve reward quality **and** cut sandbox
load, then the benchmark was rebuilt to measure the result honestly.

- **Binary all-must-pass verdict.** The committed routing scored with
  `compute_score(continuous=True)` = pass-rate over the first ≤10 cases (and a
  half-finished local edit, `res = float(res > 0)`, would have `TypeError`d on
  the returned `(score, metadata)` tuple and otherwise meant "passes ≥1 case").
  The intended verdict is binary: `reward_score/__init__.py` now uses
  `continuous=False`, and `sandbox_fusion.compute_score` returns
  `1.0 iff every test case passes else 0.0`. All test cases are run by default
  (not just 10).

- **Short-circuit cancellation.** In binary mode `check_correctness` stops at the
  first non-pass: a `threading.Event` (checked before a case queues for and
  before it spends a `max_concurrent` slot) plus `executor.shutdown(
  cancel_futures=True)` drop the remaining cases. A wrong solution no longer runs
  every test case; a correct one still does. Because a step queues ~10k cases
  behind 256 slots, most cancelled cases are still *queued* when the verdict is
  settled — so the load saved is large precisely in the saturated regime that
  caused the timeouts.

- **`bench_concurrency.py` rewritten** to drive verl's *actual* reward client
  (binary verdict + short-circuit + retry/backoff + the output matcher). It
  reproduces one step's reward phase — `train_batch_size × rollout.n` samples
  split across `reward.num_workers` workers, each with its own
  `Semaphore(max_concurrent)` (server-facing ceiling `num_workers × max_concurrent`)
  — fired as a single burst, and reports cases-actually-run (vs short-circuit
  savings), the 503/504/timeout/conn histogram, latency percentiles, and the
  worst single-sample reward time against the NCCL watchdog, while sweeping
  `max_concurrent`. The workload mix and cases-per-sample are CLI knobs; calibrate
  them from a real run's per-case reward metadata `status` histogram.

**Benchmark result (Slurm job 148631).** Run against the standing 48-slot server,
calibrated to Code-Contests-O (≈40 test cases/problem, p10–p90 18–65; ~3.3 KB
stdin/case), sweeping `max_concurrent` for 8 reward workers under two mixes
(early-RL mostly-failing, and a worst-case mostly-passing where most samples run
all their cases):

| `max_concurrent` | ceiling (×8) | cases/s (worst-case mix) | read-timeouts/step | worst-sample |
|---:|---:|---:|---:|---:|
| 4 | 32 | 68 | 0 | 74 s |
| **6** | **48** | **76** | **0** | **83 s** |
| 8 | 64 | 68 | 0 | 101 s |
| 16 | 128 | 59 | 0 | 173 s |
| 24 | 192 | 51 | 0 | 260 s |
| 32 | 256 | 42 | **45** | 390 s |

Throughput **peaks at a server-facing ceiling ≈ the server's slot count and
*degrades* above it** — the server is CPU-throughput-bound, so oversubscribing
the 48 slots / 96 cores just slows every execution; read-timeouts appear only at
ceiling ≥ 192. The optimum is **`reward.sandbox_fusion.max_concurrent=6`**
(8 × 6 = 48, matching server slots): peak throughput, lowest tail latency,
worst-sample well under the watchdog, and a 4× margin to the first timeout under
*both* mixes. Higher values (the old 16/32) sit on the downslope and are what
produced the original storms — more client concurrency was the cause, not the
cure. `verl/recipes/run_cco.sh` is set to 6. (Re-run when the server's core count
or the dataset's case-count/timeout profile changes — those move the optimum.)

## 7. Summary of changed files

| File | Change |
|------|--------|
| `sandbox/configs/run_config.py` | add `bindroot` to the `isolation` literal; add `queue_timeout` + `SANDBOX_QUEUE_TIMEOUT` env override (§6.4) |
| `sandbox/server/sandbox_api.py` | `_acquire_slot` admission control — fast 503 when no slot within `queue_timeout` (§6.4) |
| `sandbox/tests/server/test_admission.py` | unit tests for `_acquire_slot` (§6.4) |
| `verl/.../sandbox_fusion/utils.py` | retry redesign: 503 fast-retry + jittered capped backoff, replacing the 30/60/90/120 s schedule (§6.4); layout-preserving output matcher; short-circuit cancellation in `check_correctness` (§6.5) |
| `verl/.../reward_score/__init__.py` | route code rewards through the binary `continuous=False` verdict; drop the broken `float(res > 0)` (§6.5) |
| `verl/.../sandbox_fusion/__init__.py` | `continuous=False` → binary all-must-pass score; pass `short_circuit` to `check_correctness` (§6.5) |
| `verl/.../sandbox_fusion/test_output_match.py`, `test_short_circuit.py` | regression tests for the matcher and short-circuit/binary verdict (§6.5) |
| `scripts/bench_concurrency.py` | rewritten to drive verl's real reward client and replay a true per-step burst; reports short-circuit savings, 503/timeout histogram, worst-sample vs NCCL watchdog (§6.5) |
| `verl/recipes/run_cco.sh` | `actor_rollout_ref.nccl_timeout=1800` watchdog margin (§6.4) |
| `sandbox/configs/docker_bindroot.yaml` | new server config (`isolation: bindroot`) |
| `sandbox/runners/isolation.py` | `tmp_bindroot`, `build_bindroot_wrapper`, `OBJ_TMPDIR` host-backed compiler scratch, `--make-rprivate` hygiene, orphan/signal cleanup for bindroot dirs; `tmp_cpuset` per-exec core leasing |
| `sandbox/runners/base.py` | `run_commands` bindroot branch; `_build_cmd` exports `TMPDIR=$OBJ_TMPDIR`; rlimit/taskset/netns/PID-ns stand-ins (§3); stdin-drain timeout fix (§6.2) |
| `sandbox/runners/major.py` | `rlimit_as=False` opt-outs for tsx/.NET |
| `conftest.py` | `--sandbox-backend`/`--sandbox-mode`; Apptainer server launcher; `--cleanenv --fakeroot --ignore-fakeroot-command --no-home` |
| `Makefile` | `pull-apptainer-images`, `start-apptainer-container`, `test-apptainer-bindroot`, `test-rl-load`; launch flags as above |
| `sandbox/utils/logging.py` | server log level configurable via `SANDBOX_LOG_LEVEL` |
| `sandbox/tests/test_rl_load.py` | RL-load regression suite (§6.3), `stress` marker |
| `sandbox/tests/runners/test_isolation.py` | egress test skipped under bindroot + egress-blocked assertion; port-conflict test un-skipped (per-exec netns) |
| `sandbox/tests/runners/test_lean.py` | `test_lean_error` run timeout 10 → 30 |
| `sandbox/tests/runners/test_rust.py` | `test_rust_timeout` compile timeout 1 → 20 |
| `sandbox/tests/runners/test_python.py` | skip absolute-path fetch test under `bindroot` (only the cwd is exposed, like `full`) |
