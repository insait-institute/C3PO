#!/usr/bin/env python
"""Aggregate the multi-seed error-bar sweep from ON-DISK eval gens.

Source of truth is <SAVE_ROOT>/<EXPNAME>/eval_gens/<step>.jsonl, NOT wandb: the
wandb logging process crashed early on several runs while training + eval-dump
continued to completion, so wandb histories are truncated and unreliable. Each
jsonl row is one sampled completion with an `acc` field; there is no data_source
column, so the benchmark is recovered by record order against the eval parquet
(record i -> prompt i//N -> parquet.data_source[i//N]), validated to reproduce
the wandb `val-core/avg/acc/mean@8` numbers exactly.

Metric: per benchmark, mean@N over prompts (== mean of acc within the source
since every prompt has N samples); `avg` = equal-weight MACRO mean over the six
benchmarks (this matches wandb `val-core/avg`, which is NOT the micro mean).

Comparison is seed-PAIRED: seed i of each method shares data.seed (prompt shuffle
+ mini-batch order), so we report D_i = C3PO_i - GRPO_i and a paired test, which
has far more power than eyeballing two overlapping marginal std bands.

Usage:
  python scripts/aggregate_seed_sweep.py                      # final common step
  python scripts/aggregate_seed_sweep.py --mode lastk --k 3
  python scripts/aggregate_seed_sweep.py --at-step 532 --out sweep.csv
"""

import argparse
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd

SAVE_ROOT = os.path.expanduser(os.environ.get("SAVE_ROOT", "/mnt/vast/workspaces/BayesRL/bayesrl"))
EVAL_PARQUET = os.path.join(os.path.dirname(__file__), "..", "data", "math_evals.parquet")
N_SAMPLES = 8  # val_kwargs.n
ACC_FIELD = "acc"
# equal-weight macro over exactly these six sources = wandb val-core/avg
SOURCES = ["math500", "minervamath", "amc23", "aime_2024", "aime25", "aime26"]

# method -> (dir substrings that must be present, substrings that must be absent).
# Matches the run-dir names run_rl.sh emits under SAVE_ROOT.
METHODS = OrderedDict([
    ("grpo_adamw", (["olmo3_nmtron", "dapomath", "-adamw-"], ["C3PO", "M3PO", "seed_"])),
    ("c3po", (["olmo3_nmtron", "dapomath", "-ivon-", "C3PO_N2", "ESS1e9"], ["ISONOISE", "noRC", "noRS"])),
    # M3PO: M=4 posterior samples x GROUP_SIZE=4 rollouts = 16 rollouts/prompt, i.e.
    # sample-budget-matched to the GS_16 GRPO baseline.
    ("m3po", (["olmo3_nmtron", "dapomath", "-ivon-", "M3PO_M4", "GS_4", "ESS1e9"], ["ISONOISE", "noRC", "noRS"])),
])
BASELINE = "grpo_adamw"

# Original paper runs (no -seedN tag) -> treated as seed label 0. Values may be
# absolute paths (paper runs live under a different root) or names under SAVE_ROOT.
# NB old naming: 'interleave2'==C3PO_N2, 'seqmis'==seqmiss; IVONINIT_trained per the
# paper's headline C3PO (user confirmed same method as the scratch-init seeded runs).
_PAPER = "/mnt/vast/workspaces/BayesRL/bayesrl_paper_exps"
ORIGINALS = {
    "grpo_adamw": f"{_PAPER}/olmo3_nmtron-base-adamw-dapomath-LR_1.0e-6-GS_16",
    "c3po": f"{_PAPER}/olmo3_nmtron-base-ivon-dapomath-LR_1.0-GS_16-ESS1e9-IVONINIT_trained-interleave2-seqmis",
    # old naming: 'MCSAMPLES4'==M3PO_M4
    "m3po": f"{_PAPER}/olmo3_nmtron-base-ivon-dapomath-LR_1.0-GS_4-ESS1e9-IVONINIT_trained-MCSAMPLES4",
}

SEED_RE = re.compile(r"-seed(\d+)$")


def method_of(dirname):
    for m, (must, must_not) in METHODS.items():
        if all(t in dirname for t in must) and not any(t in dirname for t in must_not):
            return m
    return None


def discover_runs():
    """Return {method: {seed_label: run_dir}} from SAVE_ROOT + ORIGINALS."""
    out = defaultdict(dict)
    for name in sorted(os.listdir(SAVE_ROOT)):
        full = os.path.join(SAVE_ROOT, name)
        if not os.path.isdir(full):
            continue
        sm = SEED_RE.search(name)
        if not sm:
            continue  # seeded runs only via discovery; originals added explicitly
        m = method_of(name)
        if m is None:
            continue
        out[m][int(sm.group(1))] = full
    for m, dirname in ORIGINALS.items():
        full = dirname if os.path.isabs(dirname) else os.path.join(SAVE_ROOT, dirname)
        if os.path.isdir(full):
            out[m][0] = full  # seed label 0 = original paper run
        else:
            print(f"[warn] ORIGINALS[{m}] dir not found: {full}", file=sys.stderr)
    return out


def eval_steps(run_dir):
    d = os.path.join(run_dir, "eval_gens")
    if not os.path.isdir(d):
        return {}
    steps = {}
    for fn in os.listdir(d):
        if fn.endswith(".jsonl") and fn[:-6].isdigit():
            steps[int(fn[:-6])] = os.path.join(d, fn)
    return steps


def metrics_at_step(jsonl_path, src_order, cache):
    """Per-source mean@N and macro avg for one eval dump, with on-disk caching."""
    if jsonl_path in cache:
        return cache[jsonl_path]
    accs = []
    with open(jsonl_path) as fh:
        for line in fh:
            if line.strip():
                accs.append(json.loads(line)[ACC_FIELD])
    accs = np.asarray(accs, dtype=float)
    nprompt = len(accs) // N_SAMPLES
    if nprompt == 0:
        return None
    if nprompt != len(src_order):
        print(f"[warn] {jsonl_path}: {nprompt} prompts != {len(src_order)} parquet rows; "
              f"mapping first {min(nprompt, len(src_order))}", file=sys.stderr)
    ds = np.repeat(src_order[:nprompt], N_SAMPLES)[: len(accs)]
    per = {}
    for s in SOURCES:
        m = ds == s
        per[s] = float(accs[: len(ds)][m].mean()) if m.any() else float("nan")
    per["avg"] = float(np.nanmean([per[s] for s in SOURCES]))
    cache[jsonl_path] = per
    return per


def extract(run_dir, mode, k, at_step, src_order, cache):
    """Return (chosen_step, {metric: value}) for one run under the given rule."""
    steps = eval_steps(run_dir)
    if not steps:
        return None, None
    avail = sorted(steps)
    if at_step is not None:
        if at_step not in steps:
            return None, None
        return at_step, metrics_at_step(steps[at_step], src_order, cache)
    if mode == "final":
        s = avail[-1]
        return s, metrics_at_step(steps[s], src_order, cache)
    # lastk: mean of the last k available eval dumps
    chosen = avail[-k:]
    dicts = [metrics_at_step(steps[s], src_order, cache) for s in chosen]
    keys = ["avg"] + SOURCES
    agg = {key: float(np.mean([d[key] for d in dicts if d])) for key in keys}
    return chosen[-1], agg


def paired_stats(deltas):
    d = np.asarray([x for x in deltas if x is not None and not np.isnan(x)], dtype=float)
    n = len(d)
    if n == 0:
        return None
    mean = float(np.mean(d))
    sem = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    n_pos = int(np.sum(d > 0))
    wp = tp = float("nan")
    if n > 1:
        from scipy import stats
        try:
            wp = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            wp = float("nan")
        tp = float(stats.ttest_1samp(d, 0.0).pvalue)
    return mean, sem, wp, tp, n_pos, n


def main():
    global SAVE_ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["final", "lastk"], default="final")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--at-step", type=int, default=None,
                    help="extract all runs at this exact eval step (most comparable)")
    ap.add_argument("--require-step", type=int, default=None,
                    help="drop runs whose max eval step < this (keeps only complete runs; "
                         "use with --mode lastk so last-K is over comparable end-of-training evals)")
    ap.add_argument("--save-root", default=SAVE_ROOT)
    ap.add_argument("--out", default=None, help="CSV of per-(method,seed) metrics")
    args = ap.parse_args()

    SAVE_ROOT = os.path.expanduser(args.save_root)

    src_order = pd.read_parquet(EVAL_PARQUET)["data_source"].to_numpy()
    runs = discover_runs()
    if not runs:
        print(f"No runs found under {SAVE_ROOT}", file=sys.stderr)
        sys.exit(1)

    cache = {}
    # method -> seed -> (step, metrics)
    data = defaultdict(dict)
    print(f"SAVE_ROOT={SAVE_ROOT}   mode={args.mode}"
          + (f" k={args.k}" if args.mode == "lastk" else "")
          + (f"  at-step={args.at_step}" if args.at_step is not None else "") + "\n")
    print("Discovered runs (method / seed / max eval step / chosen step):")
    for m in METHODS:
        for seed in sorted(runs.get(m, {})):
            rd = runs[m][seed]
            steps = eval_steps(rd)
            maxs = max(steps) if steps else None
            if args.require_step is not None and (maxs is None or maxs < args.require_step):
                print(f"  [{m:11s} seed {seed}] max_step={maxs}  -> DROPPED (< require-step {args.require_step})")
                continue
            chosen, mets = extract(rd, args.mode, args.k, args.at_step, src_order, cache)
            if mets is None:
                print(f"  [{m:11s} seed {seed}] max_step={maxs}  -> NO DATA at requested step  ({os.path.basename(rd)})")
                continue
            data[m][seed] = (chosen, mets)
            flag = "" if (args.at_step is None or chosen == args.at_step) else " (!)"
            print(f"  [{m:11s} seed {seed}] max_step={maxs}  chosen={chosen}{flag}  avg={mets['avg']:.4f}")
    print()

    # Comparability guard: are all chosen steps equal?
    chosen_steps = {(m, s): v[0] for m, sd in data.items() for s, v in sd.items()}
    uniq = set(chosen_steps.values())
    if len(uniq) > 1:
        print(f"[WARN] runs were read at DIFFERENT steps {sorted(uniq)} -- 'final' numbers are not "
              f"directly comparable. Re-run with --at-step <common step> for a fair comparison.\n")

    if args.out:
        import csv
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "seed", "step", "metric", "value"])
            for m in data:
                for s in sorted(data[m]):
                    step, mets = data[m][s]
                    for key in ["avg"] + SOURCES:
                        w.writerow([m, s, step, key, mets[key]])
        print(f"Wrote per-run metrics -> {args.out}\n")

    # Report: for each non-baseline method, per-source + avg paired vs baseline.
    report_keys = ["avg"] + SOURCES
    hdr = f"{'metric':14s} {'baseline μ±σ':>18s} {'method μ±σ':>18s} {'paired Δ (μ±SEM)':>22s} {'signif':>24s}"
    for method in [m for m in METHODS if m != BASELINE]:
        seeds = sorted(set(data.get(BASELINE, {})) & set(data.get(method, {})))
        print("=" * len(hdr))
        print(f"{method}  vs  {BASELINE}    (paired seeds: {seeds})")
        print("=" * len(hdr))
        print(hdr)
        print("-" * len(hdr))
        for key in report_keys:
            bvals = np.array([data[BASELINE][s][1][key] for s in seeds], float)
            mvals = np.array([data[method][s][1][key] for s in seeds], float)
            if len(seeds) == 0:
                continue
            deltas = mvals - bvals
            st = paired_stats(deltas)
            bstd = bvals.std(ddof=1) if len(bvals) > 1 else 0.0
            mstd = mvals.std(ddof=1) if len(mvals) > 1 else 0.0
            bstr = f"{bvals.mean():.4f}±{bstd:.4f}"
            mstr = f"{mvals.mean():.4f}±{mstd:.4f}"
            if st:
                mean, sem, wp, tp, n_pos, n = st
                dstr = f"{mean:+.4f}±{sem:.4f}"
                sstr = f"{n_pos}/{n}>0  W={wp:.3f} t={tp:.3f}"
            else:
                dstr, sstr = "n/a", "n/a"
            print(f"{key:14s} {bstr:>18s} {mstr:>18s} {dstr:>22s} {sstr:>24s}")
        print()

    print("Notes:")
    print("  * μ±σ columns are ACROSS seeds (marginal bands, may overlap). Δ is the seed-PAIRED gain.")
    print("  * signif: 'a/n>0'=seeds with positive Δ; W=Wilcoxon signed-rank p; t=paired t p.")
    print("  * n=3 -> Wilcoxon floor p=0.25; lead with Δ + sign consistency + t, not W.")
    print("  * If chosen steps differ across runs, re-run with --at-step for a fair comparison.")


if __name__ == "__main__":
    main()
