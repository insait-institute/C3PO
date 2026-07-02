#!/usr/bin/env python3
"""Replay one verl RL reward-phase *step* against a live SandboxFusion server.

This drives verl's ACTUAL reward client
(``verl.utils.reward_score.sandbox_fusion``), so the benchmark reflects exactly
what training does, not a toy approximation:

  * the binary all-must-pass verdict (a solution scores 1.0 iff every test
    case passes), and
  * short-circuit cancellation -- the first failing case settles the verdict
    and the remaining cases are dropped before they hit the sandbox, and
  * the jittered retry/backoff on 503/504/timeout, and
  * the competitive-programming output matcher.

It reproduces the structure of a single training step's reward phase:

    n_samples = train_batch_size * rollout.n          # responses scored / step
    split across reward.num_workers reward workers     # each its OWN semaphore
    each worker scores its shard concurrently          # asyncio.gather in prod
    each sample fans its test cases out under that      # ThreadPoolExecutor,
        worker's Semaphore(max_concurrent)              #   gated by the semaphore

against a server with SANDBOX_MAX_CONCURRENCY execution slots.  The server-facing
concurrency ceiling is therefore ``num_workers * max_concurrent`` (e.g. 8*32=256),
which is what actually collides with the server's ~48 slots -- the old version
fired one independent request per "sample", capped levels at 96, and drained
between levels, so it under-predicted real load by ~10-40x and never exercised
short-circuit at all.

Unlike the old version this fires a whole step's worth of samples *at once* (the
real burst) and sweeps the real knob, ``reward.sandbox_fusion.max_concurrent``.

CALIBRATION.  Two inputs determine load and the bench cannot know them a priori:
the workload mix (fraction correct / wrong / TLE / ...) and the number of test
cases per problem.  Derive them from a real run: the reward metadata now carries
a per-case ``status`` (success / wrong_answer / timeout / runtime_error /
compile_error / cancelled).  The defaults below approximate a 7B distill early in
RL on Code-Contests (mostly-failing, tens of cases per problem); pass --mix and
--cases-per-sample to match your run.

Usage (against a running server):
    PYTHONUNBUFFERED=1 python scripts/bench_concurrency.py \
        --url http://localhost:8080/run_code \
        --train-batch-size 32 --rollout-n 16 --num-workers 8 \
        --max-concurrent 16,24,32,48
"""

import argparse
import os
import random
import statistics
import string
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def _add_verl_to_path(explicit: str | None):
    """Make ``import verl`` work so the bench drives the real reward client.

    verl and SandboxFusion are sibling repos under the workspace root, so the
    verl package dir is ../../verl relative to this script.  Override with
    --verl-path or VERL_PATH.
    """
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.getenv("VERL_PATH"):
        candidates.append(os.getenv("VERL_PATH"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.normpath(os.path.join(here, "..", "..", "verl")))
    for c in candidates:
        if c and os.path.isdir(os.path.join(c, "verl", "utils", "reward_score")):
            sys.path.insert(0, c)
            return c
    sys.exit(
        "Could not locate the verl package. Pass --verl-path /path/to/verl "
        "(the dir that contains the 'verl/' package), or set VERL_PATH.\n"
        f"Tried: {candidates}"
    )


# ----------------------------------------------------------------------------
# Instrumentation: count every sandbox call and HTTP attempt with no changes to
# the reward client.  call_sandbox_api is invoked once per *unique test case*
# that actually runs (so short-circuited cases are not counted -> we can measure
# the savings); requests.post fires once per HTTP attempt (so retries show up,
# and we capture the 503/504/timeout status histogram + latency).
# ----------------------------------------------------------------------------
_lock = threading.Lock()
_stats = {}


def _reset_stats():
    with _lock:
        _stats.clear()
        _stats.update(
            cases_run=0,           # call_sandbox_api invocations (cases not short-circuited)
            http_attempts=0,       # requests.post calls (includes retries)
            http_status={},        # status_code -> count (200/503/504/...)
            http_timeout=0,        # read timeouts
            http_conn=0,           # connection errors
            http_lat=[],           # per-HTTP-attempt latency
        )


def _install_instrumentation(sandbox_fusion_utils):
    orig_call = sandbox_fusion_utils.call_sandbox_api

    def counting_call(*a, **k):
        with _lock:
            _stats["cases_run"] += 1
        return orig_call(*a, **k)

    sandbox_fusion_utils.call_sandbox_api = counting_call

    orig_post = requests.post

    def counting_post(*a, **k):
        t0 = time.monotonic()
        try:
            resp = orig_post(*a, **k)
            dt = time.monotonic() - t0
            with _lock:
                _stats["http_attempts"] += 1
                _stats["http_lat"].append(dt)
                sc = resp.status_code
                _stats["http_status"][sc] = _stats["http_status"].get(sc, 0) + 1
            return resp
        except requests.exceptions.Timeout:
            with _lock:
                _stats["http_attempts"] += 1
                _stats["http_timeout"] += 1
                _stats["http_lat"].append(time.monotonic() - t0)
            raise
        except requests.exceptions.ConnectionError:
            with _lock:
                _stats["http_attempts"] += 1
                _stats["http_conn"] += 1
                _stats["http_lat"].append(time.monotonic() - t0)
            raise

    requests.post = counting_post


# ----------------------------------------------------------------------------
# Workload: build n_samples = (completion, test_cases, outcome_label) triples.
# Each outcome is engineered so the sandbox produces a deterministic verdict, so
# short-circuit behaves exactly as it will in training.
# ----------------------------------------------------------------------------
OUTCOMES = ("correct", "wrong_all", "wrong_edge", "runtime_error", "tle", "compile_error")
DEFAULT_MIX = {
    # ~7B distill early in RL on Code-Contests: mostly failing, fails early.
    "correct": 0.10,
    "wrong_all": 0.45,      # wrong on every case -> short-circuits after ~1 case
    "wrong_edge": 0.15,     # passes most, fails a late case -> runs nearly all
    "runtime_error": 0.12,
    "tle": 0.10,            # infinite loop -> holds a slot for the full timeout
    "compile_error": 0.08,
}


def _salt(rng):
    return "".join(rng.choices(string.ascii_lowercase, k=8))


def _make_sample(rng, outcome, n_cases, correct_cpu_ms, stdin_bytes):
    """Return (completion, test_cases_dict, outcome).

    The program adds the first two ints of stdin; ``stdin_bytes`` pads each case
    with trailing zeros the program ignores, so payload/IO size is modelled
    realistically while the verdict stays deterministic.  Code / expected outputs
    are varied to force the outcome."""
    inputs, outputs = [], []
    for _ in range(n_cases):
        a, b = rng.randint(0, 10**6), rng.randint(0, 10**6)
        s = f"{a} {b}"
        if stdin_bytes > len(s):
            s = s + " " + ("0 " * ((stdin_bytes - len(s)) // 2))
        inputs.append(s + "\n")
        outputs.append(str(a + b))

    busy = (
        f"_n=0\nfor _i in range(int({correct_cpu_ms} * 3000)):\n    _n+=_i\n"
        if correct_cpu_ms > 0
        else ""
    )
    # Read the WHOLE buffer (models the real per-case stdin transfer) but use
    # only the first two ints, so the zero-padding above is ignored.
    read = "import sys\n_d=sys.stdin.buffer.read().split()\na,b=int(_d[0]),int(_d[1])\n"

    if outcome == "correct":
        code = f"# {_salt(rng)}\n{read}{busy}print(a+b)"
    elif outcome == "wrong_all":
        # Unambiguously wrong on every case: *2+1 is ~100% off, so it fails the
        # matcher's float tolerance regardless of magnitude (an off-by-one on a
        # ~1e6 value is *within* rel_tol=1e-5 and would be judged correct).
        code = f"# {_salt(rng)}\n{read}print((a+b)*2+1)"
    elif outcome == "wrong_edge":
        code = f"# {_salt(rng)}\n{read}{busy}print(a+b)"  # right code; corrupt the LAST expected output
        if outputs:
            outputs[-1] = str(int(outputs[-1]) * 2 + 1)
    elif outcome == "runtime_error":
        code = f"# {_salt(rng)}\n{read}print(a//0)"
    elif outcome == "tle":
        code = f"# {_salt(rng)}\nwhile True:\n    pass"
    elif outcome == "compile_error":
        code = f"# {_salt(rng)}\ndef oops(:\n    pass"
    else:
        raise ValueError(outcome)

    completion = f"```python\n{code}\n```"
    return completion, {"inputs": inputs, "outputs": outputs}, outcome


def build_samples(n_samples, mix, cases_lo, cases_hi, correct_cpu_ms, stdin_bytes, seed):
    rng = random.Random(seed)
    labels = list(mix.keys())
    weights = list(mix.values())
    samples = []
    for _ in range(n_samples):
        outcome = rng.choices(labels, weights=weights, k=1)[0]
        n_cases = rng.randint(cases_lo, cases_hi)
        samples.append(_make_sample(rng, outcome, n_cases, correct_cpu_ms, stdin_bytes))
    return samples


# ----------------------------------------------------------------------------
# Driver: replay one step at a given max_concurrent.
# ----------------------------------------------------------------------------
def run_step(samples, compute_score, url, num_workers, max_concurrent, timeout, mem, inflight_samples):
    """Fire a whole step's worth of samples at once.

    Each sample is pinned to one of ``num_workers`` worker semaphores
    (round-robin) so the server-facing concurrency ceiling is exactly
    num_workers * max_concurrent, as in training. ``inflight_samples`` bounds
    how many sample-runner threads exist at once -- the semaphores, not this,
    are the real throttle; it just keeps local thread count sane.
    """
    sems = [threading.Semaphore(max_concurrent) for _ in range(num_workers)]
    per_sample = []  # (outcome, score, wall_seconds)

    def score_one(idx_sample):
        idx, (completion, test_cases, outcome) = idx_sample
        sem = sems[idx % num_workers]
        t0 = time.monotonic()
        score, _meta = compute_score(
            url, sem, mem, completion, test_cases, continuous=False, timeout=timeout, retry_on_timeout=True
        )
        return outcome, float(score), time.monotonic() - t0

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=inflight_samples) as pool:
        futs = [pool.submit(score_one, it) for it in enumerate(samples)]
        for f in as_completed(futs):
            per_sample.append(f.result())
    wall = time.monotonic() - t0
    return per_sample, wall


def pct(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080/run_code")
    ap.add_argument("--verl-path", default=None, help="Dir containing the verl package (default: ../../verl)")
    # Step shape -- mirror the recipe (run_cco.sh).
    ap.add_argument("--train-batch-size", type=int, default=32)
    ap.add_argument("--rollout-n", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=8, help="reward.num_workers (one semaphore each)")
    ap.add_argument("--max-concurrent", default="16,24,32,48", help="per-worker semaphore sizes to sweep")
    ap.add_argument("--run-timeout", type=int, default=10, help="per-case compile/run timeout (compute_score timeout)")
    ap.add_argument("--memory-limit-mb", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1, help="reward steps to replay per level (averaged)")
    # Workload calibration.
    ap.add_argument("--cases-per-sample", default="18,65", help="lo,hi range of test cases per problem (CCO p10-p90)")
    ap.add_argument("--stdin-bytes", type=int, default=3000, help="per-case stdin payload size (CCO median ~3.3KB)")
    ap.add_argument("--correct-cpu-ms", type=float, default=50.0, help="rough CPU cost of a correct case (busy loop)")
    ap.add_argument("--mix", default=None, help="outcome weights, e.g. correct=0.1,wrong_all=0.45,tle=0.1,...")
    ap.add_argument("--inflight-samples", type=int, default=0, help="0 = auto (num_workers * max(max_concurrent))")
    ap.add_argument("--nccl-timeout", type=float, default=1800.0, help="watchdog to compare worst sample time against")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _add_verl_to_path(args.verl_path)
    from verl.utils.reward_score.sandbox_fusion import compute_score
    from verl.utils.reward_score.sandbox_fusion import utils as sf_utils

    _install_instrumentation(sf_utils)
    _reset_stats()  # initialise before the warm-up call touches the counters

    levels = [int(x) for x in args.max_concurrent.split(",")]
    cases_lo, cases_hi = (int(x) for x in args.cases_per_sample.split(","))
    n_samples = args.train_batch_size * args.rollout_n

    mix = dict(DEFAULT_MIX)
    if args.mix:
        mix = {}
        for kv in args.mix.split(","):
            k, v = kv.split("=")
            mix[k.strip()] = float(v)
        bad = set(mix) - set(OUTCOMES)
        if bad:
            sys.exit(f"unknown outcome(s) in --mix: {bad}; valid: {OUTCOMES}")
    total = sum(mix.values())
    mix = {k: v / total for k, v in mix.items()}  # normalise

    samples = build_samples(n_samples, mix, cases_lo, cases_hi, args.correct_cpu_ms, args.stdin_bytes, args.seed)
    total_possible_cases = sum(len(tc["inputs"]) for _c, tc, _o in samples)
    print(
        f"step shape: {n_samples} samples ({args.train_batch_size} prompts x {args.rollout_n}) "
        f"across {args.num_workers} workers; {total_possible_cases} test cases if none short-circuit",
        flush=True,
    )
    print("mix: " + ", ".join(f"{k}={v:.2f}" for k, v in mix.items()), flush=True)

    # Warm-up: first executions pay one-time runtime cache costs.
    print("warming up...", flush=True)
    try:
        compute_score(args.url, threading.Semaphore(4), args.memory_limit_mb,
                      "```python\nprint(1)\n```", {"inputs": ["\n"], "outputs": ["1"]},
                      continuous=False, timeout=args.run_timeout)
    except Exception as e:  # noqa: BLE001
        print(f"  warm-up call failed ({e}); is the server up at {args.url}?", flush=True)

    results = []
    for level in levels:
        inflight = args.inflight_samples or args.num_workers * level
        # Average over --steps replays so a single unlucky burst doesn't decide.
        agg = {"wall": [], "cases_run": [], "http_attempts": [], "scores": [],
               "worst_sample": [], "status": {}, "timeouts": 0, "conn": 0, "lat": []}
        for _ in range(args.steps):
            _reset_stats()
            per_sample, wall = run_step(
                samples, compute_score, args.url, args.num_workers, level,
                args.run_timeout, args.memory_limit_mb, inflight,
            )
            with _lock:
                agg["cases_run"].append(_stats["cases_run"])
                agg["http_attempts"].append(_stats["http_attempts"])
                agg["timeouts"] += _stats["http_timeout"]
                agg["conn"] += _stats["http_conn"]
                agg["lat"].extend(_stats["http_lat"])
                for sc, c in _stats["http_status"].items():
                    agg["status"][sc] = agg["status"].get(sc, 0) + c
            agg["wall"].append(wall)
            agg["scores"].extend(s for _o, s, _w in per_sample)
            agg["worst_sample"].append(max((w for _o, _s, w in per_sample), default=0.0))

        cases_run = statistics.mean(agg["cases_run"])
        wall = statistics.mean(agg["wall"])
        saved = 100.0 * (1 - cases_run / total_possible_cases) if total_possible_cases else 0.0
        n503 = agg["status"].get(503, 0)
        n504 = agg["status"].get(504, 0)
        rec = {
            "level": level,
            "inflight_ceiling": args.num_workers * level,
            "cases_run": cases_run,
            "saved_pct": saved,
            "http_attempts": statistics.mean(agg["http_attempts"]),
            "wall": wall,
            "cases_per_s": cases_run / wall if wall else float("nan"),
            "p50": pct(agg["lat"], 50),
            "p90": pct(agg["lat"], 90),
            "p99": pct(agg["lat"], 99),
            "max": max(agg["lat"], default=float("nan")),
            "n503": n503 / args.steps,
            "n504": n504 / args.steps,
            "timeouts": agg["timeouts"] / args.steps,
            "conn": agg["conn"] / args.steps,
            "worst_sample": max(agg["worst_sample"], default=0.0),
            "mean_score": statistics.mean(agg["scores"]) if agg["scores"] else float("nan"),
        }
        results.append(rec)
        print(
            f"max_concurrent {level:3d} (ceiling {rec['inflight_ceiling']:4d}): "
            f"cases_run {cases_run:7.0f} ({saved:4.1f}% saved by short-circuit)  "
            f"wall {wall:6.1f}s  {rec['cases_per_s']:6.1f} cases/s  "
            f"p99 {rec['p99']:5.2f}s  503 {rec['n503']:.0f}  to {rec['timeouts']:.0f}  "
            f"conn {rec['conn']:.0f}  worst-sample {rec['worst_sample']:6.1f}s",
            flush=True,
        )

    # Report table.
    print("\n| max_conc | ceiling | cases_run | saved% | cases/s | p50 | p90 | p99 | max | 503 | 504 | timeouts | conn | worst_sample | mean_score |")
    print("|----------|---------|-----------|--------|---------|-----|-----|-----|-----|-----|-----|----------|------|--------------|------------|")
    for r in results:
        print(
            f"| {r['level']} | {r['inflight_ceiling']} | {r['cases_run']:.0f} | {r['saved_pct']:.1f} | "
            f"{r['cases_per_s']:.1f} | {r['p50']:.2f} | {r['p90']:.2f} | {r['p99']:.2f} | {r['max']:.2f} | "
            f"{r['n503']:.0f} | {r['n504']:.0f} | {r['timeouts']:.0f} | {r['conn']:.0f} | "
            f"{r['worst_sample']:.1f} | {r['mean_score']:.3f} |"
        )

    # Recommend: best throughput among levels with no read timeouts / connection
    # errors (503s are healthy -- admission control's fast reject) AND whose
    # worst single-sample reward time stays well under the NCCL watchdog, so a
    # slow reward never stalls a rank into a collective-timeout crash.
    watchdog_budget = 0.5 * args.nccl_timeout
    clean = [
        r for r in results
        if r["timeouts"] == 0 and r["conn"] == 0 and r["worst_sample"] < watchdog_budget
    ]
    print()
    if clean:
        best = max(clean, key=lambda r: r["cases_per_s"])
        print(
            f"Recommended: reward.sandbox_fusion.max_concurrent={best['level']} "
            f"(server-facing ceiling {best['inflight_ceiling']}, {best['cases_per_s']:.1f} cases/s, "
            f"p99 {best['p99']:.2f}s, worst-sample {best['worst_sample']:.1f}s < {watchdog_budget:.0f}s watchdog budget)."
        )
    else:
        print(
            "No level stayed clean (timeouts/conn errors, or a sample exceeded the "
            f"{watchdog_budget:.0f}s watchdog budget). The server is undersized for this "
            "step; lower max_concurrent, raise server slots/CPU, or cut cases-per-sample."
        )


if __name__ == "__main__":
    main()
