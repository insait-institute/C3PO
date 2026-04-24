"""
Empirical support dynamics analysis, following Wu et al., "The Invisible Leash".

Given a base model and one or more trained models, each with `eval_correctness.pkl`
mapping prompt_id -> list of bools (one entry per sample), compute per-prompt
support classification and the aggregate metrics SRR, NDR, SDS, NSCR.

Support rule (prompt-level, paper Sec. 3.1):
    epsilon = -log(0.05) / k  with k = num samples per prompt.
    A prompt is "in the support" of a model iff the fraction of correct
    samples > epsilon, i.e. #correct / k > epsilon.

Support categories per prompt:
    P (Preservation): base in-support AND trained in-support
    S (Shrinkage):    base in-support AND trained OUT-of-support
    E (Expansion):    base OUT-of-support AND trained in-support
    O (Out):          both OUT-of-support
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# --- Configuration -----------------------------------------------------------

ROOT = Path(__file__).parents[0] / "eval_preds"
BASE_DIR = ROOT / "olmo3_nmtron_ivon"           # base model
TRAINED_DIRS = [
    ROOT / "olmo3_nmtron_adamw_LR1.0_GS16",
    ROOT / "olmo3_nmtron_ivon_LR1.0_GS16_ESS1e9",
]

CONFIDENCE = 0.05   # zeta; 95% confidence bound from paper Appx. C.4
K_EXPECTED = 4096   # samples per prompt

# --- Loading -----------------------------------------------------------------

def load_correctness(model_dir: Path) -> Dict[object, List[bool]]:
    """Load eval_correctness.pkl -> {prompt_id: [bool, ...]}."""
    path = model_dir / "eval_correctness.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected dict in {path}, got {type(data).__name__}"
        )
    return data


def correct_counts(correctness: Dict[object, List[bool]]) -> Dict[object, Tuple[int, int]]:
    """Return {prompt_id: (num_correct, num_samples)}."""
    out = {}
    for pid, flags in correctness.items():
        # Be liberal: accept anything truthy as correct.
        flags_list = list(flags)
        n = len(flags_list)
        c = sum(1 for v in flags_list if bool(v))
        out[pid] = (c, n)
    return out

# --- Metrics -----------------------------------------------------------------

def epsilon_from_k(k: int, zeta: float = CONFIDENCE) -> float:
    """Paper Appx. C.4: eps = -log(zeta)/k."""
    return -math.log(zeta) / k


def in_support(num_correct: int, k: int, eps: float) -> bool:
    """Prompt is in support if p_hat = c/k > eps."""
    if k == 0:
        return False
    return (num_correct / k) > eps


def classify_prompts(
    base_counts: Dict[object, Tuple[int, int]],
    trained_counts: Dict[object, Tuple[int, int]],
) -> Tuple[pd.DataFrame, float]:
    """
    For every prompt shared by both models, determine the support category.
    Returns a per-prompt DataFrame plus the epsilon used.
    k is read from the data itself (we use the base model's k; if the
    trained model has a different k, we note it but still compare with
    each model's own eps).
    """
    common = sorted(set(base_counts) & set(trained_counts), key=lambda p: str(p))
    missing_base = set(trained_counts) - set(base_counts)
    missing_trained = set(base_counts) - set(trained_counts)
    if missing_base or missing_trained:
        print(
            f"  note: {len(missing_base)} prompt(s) only in trained, "
            f"{len(missing_trained)} only in base; using {len(common)} common."
        )

    rows = []
    for pid in common:
        c_b, k_b = base_counts[pid]
        c_t, k_t = trained_counts[pid]
        eps_b = epsilon_from_k(k_b)
        eps_t = epsilon_from_k(k_t)
        base_in = in_support(c_b, k_b, eps_b)
        trained_in = in_support(c_t, k_t, eps_t)
        if base_in and trained_in:
            cat = "P"
        elif base_in and not trained_in:
            cat = "S"
        elif not base_in and trained_in:
            cat = "E"
        else:
            cat = "O"
        rows.append({
            "prompt_id": pid,
            "base_correct": c_b,
            "base_k": k_b,
            "base_frac": c_b / k_b if k_b else 0.0,
            "trained_correct": c_t,
            "trained_k": k_t,
            "trained_frac": c_t / k_t if k_t else 0.0,
            "category": cat,
        })

    df = pd.DataFrame(rows)
    # Report the base-k-derived eps as the headline threshold.
    # (In this study all models share k=4096, so eps_b == eps_t.)
    eps_headline = (
        epsilon_from_k(df["base_k"].iloc[0]) if len(df) else float("nan")
    )
    return df, eps_headline


def support_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Compute SRR, NDR, SDS, NSCR + raw counts from a per-prompt DataFrame."""
    counts = df["category"].value_counts().to_dict()
    P = counts.get("P", 0)
    E = counts.get("E", 0)
    S = counts.get("S", 0)
    O = counts.get("O", 0)

    srr = P / (P + S) if (P + S) > 0 else float("nan")
    ndr = E / (P + E) if (P + E) > 0 else float("nan")
    if srr != srr or ndr != ndr or (srr + ndr) == 0:  # nan or zero denom
        sds = float("nan") if (srr != srr or ndr != ndr) else 0.0
    else:
        sds = 2 * srr * ndr / (srr + ndr)
    nscr = (E - S) / (P + E + S) if (P + E + S) > 0 else float("nan")

    return {
        "P": P, "E": E, "S": S, "O": O,
        "SRR": srr, "NDR": ndr, "SDS": sds, "NSCR": nscr,
        "base_pass_any": (P + S) / max(P + E + S + O, 1),
        "trained_pass_any": (P + E) / max(P + E + S + O, 1),
    }

# --- Reporting ---------------------------------------------------------------

def fmt_metrics(name: str, m: Dict[str, float], eps: float) -> str:
    return (
        f"\n=== {name} ===\n"
        f"  epsilon (paper formula, zeta=0.05) = {eps:.6e}\n"
        f"  Counts:   P={m['P']}  E={m['E']}  S={m['S']}  O={m['O']}  "
        f"(total={m['P']+m['E']+m['S']+m['O']})\n"
        f"  SRR  = {m['SRR']:.4f}   (Support Retention)\n"
        f"  NDR  = {m['NDR']:.4f}   (Net Discovery)\n"
        f"  SDS  = {m['SDS']:.4f}   (Balanced Harmonic)\n"
        f"  NSCR = {m['NSCR']:+.4f}  (Net Support Change; +expand / -shrink)\n"
        f"  Base in-support fraction:    {m['base_pass_any']:.3f}\n"
        f"  Trained in-support fraction: {m['trained_pass_any']:.3f}\n"
    )

# --- Main --------------------------------------------------------------------

def main():
    print(f"Base model: {BASE_DIR}")
    base_corr = load_correctness(BASE_DIR)
    base_counts = correct_counts(base_corr)

    # Quick sanity: sample sizes
    ks = {k for (_, k) in base_counts.values()}
    print(f"  {len(base_counts)} prompts, sample counts seen: {sorted(ks)}")
    if K_EXPECTED not in ks:
        print(
            f"  WARNING: expected k={K_EXPECTED} but saw {sorted(ks)}. "
            f"Proceeding with actual k per prompt."
        )

    per_prompt_frames = []
    summary_rows = []

    for tdir in TRAINED_DIRS:
        print(f"\nTrained model: {tdir}")
        try:
            t_corr = load_correctness(tdir)
        except FileNotFoundError as e:
            print(f"  SKIPPING: {e}")
            continue
        t_counts = correct_counts(t_corr)
        t_ks = {k for (_, k) in t_counts.values()}
        print(f"  {len(t_counts)} prompts, sample counts seen: {sorted(t_ks)}")

        df, eps = classify_prompts(base_counts, t_counts)
        m = support_metrics(df)
        print(fmt_metrics(tdir.name, m, eps))

        df_out = df.copy()
        df_out.insert(0, "trained_model", tdir.name)
        per_prompt_frames.append(df_out)

        summary_rows.append({
            "trained_model": tdir.name,
            "P": m["P"], "E": m["E"], "S": m["S"], "O": m["O"],
            "SRR": m["SRR"], "NDR": m["NDR"],
            "SDS": m["SDS"], "NSCR": m["NSCR"],
            "base_pass_any": m["base_pass_any"],
            "trained_pass_any": m["trained_pass_any"],
            "epsilon": eps,
        })

    # Persist results
    outdir = Path(__file__).parents[0]
    outdir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = outdir / "support_dynamics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nWrote summary: {summary_path}")

    if per_prompt_frames:
        per_prompt_df = pd.concat(per_prompt_frames, ignore_index=True)
        per_prompt_path = outdir / "support_dynamics_per_prompt.csv"
        per_prompt_df.to_csv(per_prompt_path, index=False)
        print(f"Wrote per-prompt breakdown: {per_prompt_path}")

    print("\nSummary table:")
    if not summary_df.empty:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()