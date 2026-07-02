#!/usr/bin/env python3
"""Sweep sandbox concurrency *levels* against a live SandboxFusion server using
REAL, CORRECT competitive-programming solutions from ``inclusionAI/Ring-lite-rl-
data`` (the ``code_contests`` split).

Purpose: measure how robust each value of ``reward.sandbox_fusion.max_concurrent``
is to true RL reward load -- i.e. genuine, non-trivial solutions (real compile +
run cost) checked against their own validated test cases (tens of cases each),
rather than the synthetic a+b programs of ``bench_concurrency.py``.  We fire only
*correct* groundtruth solutions, so:

  * every test case actually runs (no short-circuit -- the heaviest, steady-state
    load), and
  * the expected verdict is 1.0, so the mean score / pass-rate at each level is a
    direct robustness signal: a level that saturates the server drops cases to
    read-timeouts, which fail the all-must-pass verdict and pull the score below
    1.0.  A healthy level scores ~1.0; a collapsing one does not.

CONCURRENCY MODEL.  verl wires up ONE shared ``multiprocessing.Manager().Semaphore(
reward.sandbox_fusion.max_concurrent)`` (verl/trainer/ppo/reward.py) and fans every
response in the step across it -- a single global ceiling on concurrent sandbox
case-requests.  This bench reproduces exactly that: one shared semaphore whose size
is the swept knob.  At each level we drive verl's ACTUAL reward client
(``verl.utils.reward_score.sandbox_fusion``) over a whole step's worth of correct
solutions and report throughput, latency, read-timeouts/503s, the worst single
reward time (vs the NCCL watchdog), and the achieved score.

The workload size mirrors the CCO recipe (run_cco.sh): ``train_batch_size`` x
``rollout.n`` responses (32 x 16 = 512).  Since all are correct, we draw that many
DISTINCT real problems (cycling the pool only if it is smaller) so server-side
caching of identical code does not flatter the numbers.

Usage (against a running server):
    PYTHONUNBUFFERED=1 python scripts/bench_concurrency_ringlite.py \
        --url http://localhost:8080/run_code \
        --train-batch-size 32 --rollout-n 16 \
        --levels 16,32,48,64,96,128
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DATASET_URL = (
    "https://huggingface.co/datasets/inclusionAI/Ring-lite-rl-data/resolve/main/code.jsonl"
)


def _add_verl_to_path(explicit):
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
# the reward client.  call_sandbox_api fires once per *unique test case that runs*
# (with all-correct solutions that is every case, no short-circuit); requests.post
# fires once per HTTP attempt (so retries show up, and we capture the 503/504/
# timeout histogram + latency).
# ----------------------------------------------------------------------------
_lock = threading.Lock()
_stats = {}


def _reset_stats():
    with _lock:
        _stats.clear()
        _stats.update(
            cases_run=0,
            http_attempts=0,
            http_status={},
            http_timeout=0,
            http_conn=0,
            http_lat=[],
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
# Data: load real CORRECT problems from inclusionAI/Ring-lite-rl-data code.jsonl.
# Each row carries a verified Python ``groundtruth`` and ``code_test_cases``
# (a list of {"input", "output"} stdin/stdout pairs).  We stream the file (it is
# ~200MB) only until we have a pool of qualifying problems, then cache that pool
# locally as jsonl so reruns are offline.
# ----------------------------------------------------------------------------
def _qualifies(row, dataset, min_cases, max_cases):
    if dataset and row.get("dataset") != dataset:
        return False
    if row.get("code_language") not in (None, "python", "python3"):
        return False
    if row.get("groundtruth_language") not in (None, "python", "python3"):
        return False
    code = row.get("groundtruth")
    cases = row.get("code_test_cases")
    if not code or not isinstance(cases, list) or not cases:
        return False
    # stdin/stdout cases only (skip assert/function-call style).
    if any("input" not in c or "output" not in c for c in cases):
        return False
    n = len(cases)
    if n < min_cases or (max_cases and n > max_cases):
        return False
    return True


def _slim(row, max_cases_cap):
    cases = row["code_test_cases"]
    if max_cases_cap and len(cases) > max_cases_cap:
        cases = cases[:max_cases_cap]
    return {
        "mid": row.get("mid"),
        "dataset": row.get("dataset"),
        "difficulty": row.get("difficulty"),
        "groundtruth": row["groundtruth"],
        "inputs": [c["input"] for c in cases],
        "outputs": [c["output"] for c in cases],
    }


def load_pool(cache_path, data_path, pool_size, dataset, min_cases, max_cases, max_cases_cap):
    """Return a list of slim problem dicts (groundtruth + inputs/outputs).

    Order of preference: an explicit --data-path, else a previously cached pool,
    else stream from HuggingFace and cache.
    """
    # pool_size == 0 means "every qualifying problem in the file" (full-dataset
    # audit); a positive value streams only until that many are collected.
    want_all = pool_size <= 0
    if not data_path and os.path.exists(cache_path):
        pool = []
        with open(cache_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    pool.append(json.loads(line))
        if dataset:
            pool = [p for p in pool if p.get("dataset") == dataset]
        if want_all or len(pool) >= pool_size:
            print(f"loaded {len(pool)} cached problems from {cache_path}", flush=True)
            return pool
        print(f"cache has only {len(pool)} usable problems (<{pool_size}); refilling...", flush=True)

    src = data_path or DATASET_URL
    is_url = src.startswith("http")
    target = "every" if want_all else pool_size
    print(f"scanning {'stream ' if is_url else ''}{src} for {target} correct "
          f"{dataset or 'python'} problems ({min_cases}-{max_cases or 'inf'} cases each)...", flush=True)

    pool = []
    line_iter = urllib.request.urlopen(src, timeout=120) if is_url else open(src, "rb")
    scanned = 0
    try:
        for raw in line_iter:
            scanned += 1
            try:
                row = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if _qualifies(row, dataset, min_cases, max_cases):
                pool.append(_slim(row, max_cases_cap))
                if not want_all and len(pool) >= pool_size:
                    break
    finally:
        line_iter.close()
    print(f"  scanned {scanned} rows -> kept {len(pool)} problems", flush=True)
    if not pool:
        sys.exit("No qualifying problems found; relax --dataset/--min-cases/--max-cases.")

    if not data_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            for p in pool:
                f.write(json.dumps(p) + "\n")
        print(f"  cached pool -> {cache_path}", flush=True)
    return pool


def build_workload(pool, n_samples, seed):
    """Return n_samples (completion, test_cases, mid) triples of CORRECT solutions.

    Distinct real problems are preferred; the pool is cycled only if it is smaller
    than n_samples (a warning is printed), to avoid server-side caching of
    identical code flattering the throughput.
    """
    rng = random.Random(seed)
    problems = list(pool)
    rng.shuffle(problems)  # shuffle so worker shards see a mix of easy/hard
    if n_samples <= 0:
        chosen = problems  # full-dataset audit: every problem exactly once
    else:
        if n_samples > len(problems):
            print(f"  WARNING: pool has {len(problems)} problems < {n_samples} samples; cycling "
                  "(some identical code may be cached server-side).", flush=True)
        chosen = [problems[i % len(problems)] for i in range(n_samples)]
    workload = []
    for p in chosen:
        completion = f"```python\n{p['groundtruth']}\n```"
        workload.append((completion, {"inputs": p["inputs"], "outputs": p["outputs"]}, p["mid"]))
    return workload


# ----------------------------------------------------------------------------
# Driver: replay the whole step at one concurrency level (one shared semaphore).
# ----------------------------------------------------------------------------
def run_level(workload, compute_score, url, level, timeout, mem, inflight,
              num_workers=1, batch_size=0, progress=False):
    """Score every (correct) sample at concurrency ``level``.

    ``num_workers`` semaphores of size ``level`` are created and each sample is
    pinned to one round-robin, so the server-facing ceiling is num_workers*level
    -- exactly verl's ``reward.num_workers`` x ``reward.sandbox_fusion.max_
    concurrent`` arrangement (num_workers=1 collapses to the single global
    semaphore).  When ``batch_size`` > 0 the workload is processed in sequential
    chunks of that many samples (one chunk drained before the next starts), so a
    huge workload is replayed as a sequence of RL-step-sized batches rather than
    one full burst.  ``inflight`` bounds the local sample-runner threads.
    """
    sems = [threading.Semaphore(level) for _ in range(max(1, num_workers))]
    per_sample = []  # (score, wall_seconds)

    def score_one(item, sem):
        completion, test_cases, _mid = item
        t0 = time.monotonic()
        score, _meta = compute_score(
            url, sem, mem, completion, test_cases,
            continuous=False, timeout=timeout, retry_on_timeout=True,
        )
        return float(score), time.monotonic() - t0

    n = len(workload)
    bs = batch_size if (batch_size and batch_size > 0) else n
    n_batches = math.ceil(n / bs)
    t0 = time.monotonic()
    for b, start in enumerate(range(0, n, bs)):
        chunk = list(enumerate(workload[start:start + bs], start))
        with ThreadPoolExecutor(max_workers=inflight) as pool:
            futs = [pool.submit(score_one, it, sems[gi % len(sems)]) for gi, it in chunk]
            for f in as_completed(futs):
                per_sample.append(f.result())
        if progress:
            passed = sum(1 for s, _w in per_sample if s >= 1.0)
            elapsed = time.monotonic() - t0
            cps = (_stats["cases_run"] / elapsed) if elapsed else float("nan")
            print(f"  batch {b + 1}/{n_batches}: {len(per_sample)}/{n} scored, "
                  f"pass {100 * passed / len(per_sample):.1f}%, {cps:.0f} cases/s, {elapsed:.0f}s",
                  flush=True)
    wall = time.monotonic() - t0
    return per_sample, wall


def pct(values, p):
    if not values:
        return float("nan")
    values = sorted(values)
    k = min(len(values) - 1, max(0, round(p / 100 * (len(values) - 1))))
    return values[k]


def replay_level(level, workload, total_cases, compute_score, args, inflight):
    """Replay the step --steps times at ``level`` and return an aggregate record."""
    agg = {"wall": [], "cases_run": [], "http_attempts": [], "scores": [],
           "worst_rollout": [], "status": {}, "timeouts": 0, "conn": 0, "lat": []}
    for _ in range(args.steps):
        _reset_stats()
        per_sample, wall = run_level(
            workload, compute_score, args.url, level,
            args.run_timeout, args.memory_limit_mb, inflight,
            num_workers=args.num_workers, batch_size=args.batch_size,
            progress=(args.batch_size and args.batch_size > 0),
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
        agg["scores"].extend(s for s, _w in per_sample)
        agg["worst_rollout"].append(max((w for _s, w in per_sample), default=0.0))

    cases_run = statistics.mean(agg["cases_run"])
    wall = statistics.mean(agg["wall"])
    scores = agg["scores"]
    return {
        "level": level,
        "ceiling": level * args.num_workers,
        "cases_run": cases_run,
        "http_attempts": statistics.mean(agg["http_attempts"]),
        "wall": wall,
        "cases_per_s": cases_run / wall if wall else float("nan"),
        "p50": pct(agg["lat"], 50),
        "p90": pct(agg["lat"], 90),
        "p99": pct(agg["lat"], 99),
        "max": max(agg["lat"], default=float("nan")),
        "n503": agg["status"].get(503, 0) / args.steps,
        "n504": agg["status"].get(504, 0) / args.steps,
        "timeouts": agg["timeouts"] / args.steps,
        "conn": agg["conn"] / args.steps,
        "worst_rollout": max(agg["worst_rollout"], default=0.0),
        "mean_score": statistics.mean(scores) if scores else float("nan"),
        "pass_rate": (sum(1 for s in scores if s >= 1.0) / len(scores)) if scores else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080/run_code")
    ap.add_argument("--verl-path", default=None, help="Dir containing the verl package (default: ../../verl)")
    # Step shape -- mirror the recipe (run_cco.sh): n_samples = train_batch_size * rollout.n.
    ap.add_argument("--train-batch-size", type=int, default=32)
    ap.add_argument("--rollout-n", type=int, default=16)
    ap.add_argument("--levels", default="16,32,48,64,96,128",
                    help="reward.sandbox_fusion.max_concurrent values to sweep (per-worker semaphore size)")
    ap.add_argument("--num-workers", type=int, default=1,
                    help="reward.num_workers: that many semaphores of size <level> (server ceiling = num_workers*level)")
    ap.add_argument("--batch-size", type=int, default=0,
                    help="process the workload in sequential chunks of this many samples (0 = single pass)")
    ap.add_argument("--all-problems", action="store_true",
                    help="audit EVERY qualifying problem once (ignores train-batch-size x rollout-n; scans whole file)")
    ap.add_argument("--run-timeout", type=int, default=10, help="per-case compile/run timeout (compute_score timeout)")
    ap.add_argument("--memory-limit-mb", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=1, help="reward steps to replay per level (averaged)")
    ap.add_argument("--inflight-samples", type=int, default=0,
                    help="0 = auto; local sample-runner threads (the semaphore is the real throttle)")
    ap.add_argument("--nccl-timeout", type=float, default=1800.0, help="watchdog to compare worst rollout time against")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose-client", action="store_true",
                    help="keep verl's client-side WARNING/ERROR retry logs (default: silenced to keep the log small)")
    # Data source.
    ap.add_argument("--dataset", default="code_contests", help="restrict to this source split (\"\" = any)")
    ap.add_argument("--data-path", default=None, help="local code.jsonl (default: stream from HuggingFace + cache)")
    ap.add_argument("--cache-path", default=None, help="where to cache the scanned problem pool (jsonl)")
    ap.add_argument("--pool-size", type=int, default=512, help="problems to scan/cache; must be >= n_samples for distinct")
    ap.add_argument("--min-cases", type=int, default=10, help="skip problems with fewer test cases")
    ap.add_argument("--max-cases", type=int, default=0, help="skip problems with more than this many cases (0=inf)")
    ap.add_argument("--max-cases-cap", type=int, default=100,
                    help="truncate each problem to this many cases (0=keep all; bounds local threads)")
    args = ap.parse_args()

    if not args.verbose_client:
        # verl's sandbox_fusion client logs every retry/timeout at WARNING/ERROR;
        # under a storming level that is tens of thousands of lines. Silence it --
        # the bench captures the same information in its timeout/503 counters.
        import logging
        logging.disable(logging.ERROR)

    _add_verl_to_path(args.verl_path)
    from verl.utils.reward_score.sandbox_fusion import compute_score
    from verl.utils.reward_score.sandbox_fusion import utils as sf_utils

    _install_instrumentation(sf_utils)
    _reset_stats()  # initialise before the warm-up call touches the counters

    levels = [int(x) for x in args.levels.split(",")]
    n_samples = args.train_batch_size * args.rollout_n

    here = os.path.dirname(os.path.abspath(__file__))
    default_cache = "ring_lite_code_all_python.jsonl" if args.all_problems else "ring_lite_code_pool.jsonl"
    cache_path = args.cache_path or os.path.normpath(os.path.join(here, "..", "data", default_cache))
    want_pool = 0 if args.all_problems else max(args.pool_size, n_samples)
    pool = load_pool(cache_path, args.data_path, want_pool,
                     args.dataset, args.min_cases, args.max_cases, args.max_cases_cap)

    workload = build_workload(pool, 0 if args.all_problems else n_samples, args.seed)
    total_cases = sum(len(tc["inputs"]) for _c, tc, _m in workload)
    case_counts = [len(tc["inputs"]) for _c, tc, _m in workload]
    input_bytes = [len(i) for _c, tc, _m in workload for i in tc["inputs"]]
    median_cases = max(1, int(statistics.median(case_counts)))

    # Auto inflight: enough sample threads to keep the largest ceiling's worth of
    # cases ready, but capped to bound local threads (each active compute_score
    # spawns up to ~its-case-count threads, all blocked on its worker semaphore).
    max_ceiling = max(levels) * args.num_workers
    if args.inflight_samples:
        inflight = args.inflight_samples
    else:
        inflight = max(64, max_ceiling, 2 * math.ceil(max_ceiling / max(1, min(case_counts))))
    if args.batch_size and args.batch_size > 0:
        inflight = min(inflight, args.batch_size)
    inflight = min(inflight, len(workload))
    est_threads = inflight * median_cases

    n_work = len(workload)
    if args.all_problems:
        print(f"FULL-DATASET AUDIT: {n_work} CORRECT problems (every qualifying one), "
              f"dataset={args.dataset or 'any python'}", flush=True)
    else:
        print(f"step shape: {n_work} CORRECT responses ({args.train_batch_size} prompts x {args.rollout_n}); "
              f"dataset={args.dataset or 'any'}", flush=True)
    if args.num_workers > 1:
        print(f"  concurrency: {args.num_workers} workers x level (server ceiling = {args.num_workers}*level); "
              f"levels {levels}", flush=True)
    else:
        print(f"  concurrency levels swept (shared semaphore size): {levels}", flush=True)
    if args.batch_size and args.batch_size > 0:
        print(f"  batched: {math.ceil(n_work / args.batch_size)} sequential batches of {args.batch_size}", flush=True)
    print(
        f"  test cases/problem: min {min(case_counts)} median {median_cases} max {max(case_counts)}; "
        f"{total_cases} total cases (no short-circuit -- all correct)", flush=True)
    print(
        f"  stdin bytes/case: median {int(statistics.median(input_bytes))} max {max(input_bytes)}", flush=True)
    print(f"  inflight sample threads: {inflight} (~{est_threads} peak case threads)", flush=True)
    if est_threads > 12000:
        print("  WARNING: high local thread estimate; consider --max-cases-cap or --inflight-samples.", flush=True)

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
        rec = replay_level(level, workload, total_cases, compute_score, args, inflight)
        results.append(rec)
        print(
            f"max_concurrent {level:4d} (ceiling {rec['ceiling']:4d}): {rec['cases_per_s']:6.1f} cases/s  "
            f"wall {rec['wall']:7.1f}s  p50 {rec['p50']:5.2f}s  p99 {rec['p99']:5.2f}s  503 {rec['n503']:.0f}  "
            f"to {rec['timeouts']:.0f}  conn {rec['conn']:.0f}  worst-rollout {rec['worst_rollout']:6.1f}s  "
            f"pass {100 * rec['pass_rate']:5.1f}%  score {rec['mean_score']:.3f}",
            flush=True,
        )

    # Report table.
    print("\n| max_concurrent | ceiling | cases/s | wall_s | p50 | p90 | p99 | max | 503 | 504 | timeouts | conn | worst_rollout | pass% | mean_score |")
    print("|----------------|---------|---------|--------|-----|-----|-----|-----|-----|-----|----------|------|---------------|-------|------------|")
    for r in results:
        print(
            f"| {r['level']} | {r['ceiling']} | {r['cases_per_s']:.1f} | {r['wall']:.0f} | {r['p50']:.2f} | "
            f"{r['p90']:.2f} | {r['p99']:.2f} | {r['max']:.2f} | {r['n503']:.0f} | {r['n504']:.0f} | "
            f"{r['timeouts']:.0f} | {r['conn']:.0f} | {r['worst_rollout']:.1f} | {100 * r['pass_rate']:.1f} | "
            f"{r['mean_score']:.3f} |"
        )

    # Recommend: highest-throughput level that stays robust to true RL load --
    # no read-timeouts / connection errors (503s are healthy fast-rejects), the
    # worst single reward time well under the NCCL watchdog, and corrects still
    # scoring ~1.0 (a level that drops the pass-rate is dropping real cases).
    watchdog_budget = 0.5 * args.nccl_timeout
    clean = [
        r for r in results
        if r["timeouts"] == 0 and r["conn"] == 0
        and r["worst_rollout"] < watchdog_budget and r["pass_rate"] >= 0.99
    ]
    print()
    if clean:
        best = max(clean, key=lambda r: r["cases_per_s"])
        print(
            f"Recommended: reward.sandbox_fusion.max_concurrent={best['level']} "
            f"({best['cases_per_s']:.1f} cases/s, p99 {best['p99']:.2f}s, "
            f"worst-rollout {best['worst_rollout']:.1f}s < {watchdog_budget:.0f}s watchdog budget, "
            f"pass {100 * best['pass_rate']:.1f}%).")
    else:
        print(
            "No level stayed fully robust (read-timeouts/conn errors, a reward exceeded the "
            f"{watchdog_budget:.0f}s watchdog budget, or corrects dropped below a 99% pass-rate). "
            "Lower max_concurrent, raise server slots/CPU, or cut cases-per-problem.")


if __name__ == "__main__":
    main()
