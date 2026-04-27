"""Compute rollout diversity metrics per training step.

Each rollout jsonl holds (typically) 32 unique prompts x 16 generations for
one step, with columns: input, output, gts, pred, score, step.

Metrics (per-prompt, then averaged across prompts):
  * final_answer_div  — unique values of `pred` among 16 generations.
  * EAD (lexical)     — Expectation-Adjusted Distinct n-grams over `output`,
                        averaged for n in 1..5. Liu et al. 2022.
  * SBERT (semantic)  — 1 - mean pairwise cosine sim of all-mpnet-base-v2
                        embeddings of `output`.
  * Vendi Score       — exp(entropy(eigvals(G/K))) on the SBERT cosine kernel.

Reference: arXiv:2604.16027, Appendix B.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def ngrams(tokens, n):
    return list(zip(*(tokens[i:] for i in range(n))))


def ead_for_group(outputs, tokenizer, ns=(1, 2, 3, 4, 5)):
    V = tokenizer.vocab_size
    token_seqs = [tokenizer.encode(o, add_special_tokens=False) for o in outputs]
    scores = []
    for n in ns:
        all_ngrams = []
        for seq in token_seqs:
            all_ngrams.extend(ngrams(seq, n))
        T = len(all_ngrams)
        if T == 0:
            continue
        U = len(set(all_ngrams))
        expected = V * (1 - ((V - 1) / V) ** T)
        scores.append(min(1.0, U / expected))
    return float(np.mean(scores)) if scores else float("nan")


def sbert_kernel(outputs, encoder):
    embs = encoder.encode(outputs, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    return embs @ embs.T


def sbert_for_group(G):
    K = G.shape[0]
    iu = np.triu_indices(K, k=1)
    return float(1.0 - G[iu].mean())


def vendi_for_group(G):
    K = G.shape[0]
    P = G / K
    P = (P + P.T) / 2  # symmetrize against numerical drift
    eigvals = np.linalg.eigvalsh(P)
    eigvals = np.clip(eigvals, 1e-12, None)
    eigvals = eigvals / eigvals.sum()
    H = -(eigvals * np.log(eigvals)).sum()
    return float(np.exp(H))


def final_answer_diversity(preds):
    return preds.astype(str).nunique()


def self_bleu_for_group(outputs, max_n=4):
    """Self-BLEU diversity = 1 - mean(BLEU(o_i | refs={o_j : j != i})).

    Uses NLTK corpus_bleu with weights uniform over 1..max_n. Higher = more diverse.
    """
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    tokenized = [o.split() for o in outputs]
    weights = tuple([1.0 / max_n] * max_n)
    smoothing = SmoothingFunction().method1
    scores = []
    for i, hyp in enumerate(tokenized):
        if not hyp:
            continue
        refs = [tokenized[j] for j in range(len(tokenized)) if j != i and tokenized[j]]
        if not refs:
            continue
        scores.append(sentence_bleu(refs, hyp, weights=weights, smoothing_function=smoothing))
    if not scores:
        return float("nan")
    return float(1.0 - np.mean(scores))


def metrics_for_step(df, tokenizer, encoder, want_fad, want_ead, want_sbert, want_vendi, want_self_bleu):
    rows = []
    for prompt, g in df.groupby("input", sort=False):
        outputs = g["output"].astype(str).tolist()
        rec = {}
        if want_fad:
            rec["final_answer_div"] = final_answer_diversity(g["pred"])
        if want_ead:
            rec["ead"] = ead_for_group(outputs, tokenizer)
        if want_self_bleu:
            rec["self_bleu_div"] = self_bleu_for_group(outputs)
        if want_sbert or want_vendi:
            G = sbert_kernel(outputs, encoder)
            if want_sbert:
                rec["sbert"] = sbert_for_group(G)
            if want_vendi:
                rec["vendi"] = vendi_for_group(G)
        rows.append(rec)
    return pd.DataFrame(rows).mean().to_dict()


def main(args):
    work = os.environ.get("WORK")
    print(f"Running for metrics: {args.metrics}")
    model_dir = Path(work) / "bayesrl_paper_exps" / args.model / "rollout"
    files = sorted(model_dir.glob("*.jsonl"), key=lambda p: int(p.stem))
    if not files:
        raise SystemExit(f"No jsonl files in {model_dir}")

    want_fad = "fad" in args.metrics
    want_ead = "ead" in args.metrics
    want_sbert = "sbert" in args.metrics
    want_vendi = "vendi" in args.metrics
    want_self_bleu = "self_bleu" in args.metrics

    tokenizer = None
    if want_ead:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    encoder = None
    if want_sbert or want_vendi:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(args.sbert_model, device=args.device)

    out_path = Path(args.out) if args.out else Path(__file__).parents[0] / "output_diversity.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else None

    results = []
    for f in files:
        df = pd.read_json(f, lines=True)
        step = int(df["step"].iloc[0])
        if existing is not None and not args.overwrite and ((existing["model"] == args.model) & (existing["step"] == step)).any():
            print(f"step {step:>5d}  [skip — already in {out_path.name}]")
            continue
        m = metrics_for_step(df, tokenizer, encoder, want_fad, want_ead, want_sbert, want_vendi, want_self_bleu)
        m["model"] = args.model
        m["step"] = step
        m["n_prompts"] = df["input"].nunique()
        m["acc"] = float(df["score"].mean())
        results.append(m)
        print(f"step {step:>5d}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items() if k not in ("model", "step", "n_prompts")))

    if not results:
        print(f"\nno new rows; {out_path} unchanged")
        return

    new = pd.DataFrame(results)
    cols = ["model", "step", "n_prompts", "acc"] + [c for c in ("final_answer_div", "ead", "self_bleu_div", "sbert", "vendi") if c in new.columns]
    new = new[cols]

    if existing is not None:
        merged = pd.concat([existing, new], ignore_index=True)
        merged = merged.drop_duplicates(subset=["model", "step"], keep="last")
    else:
        merged = new
    merged = merged.sort_values(["model", "step"]).reset_index(drop=True)
    merged.to_csv(out_path, index=False)
    print(f"\nwrote {out_path} ({len(new)} new rows, {len(merged)} total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model dir name under $WORK/bayesrl_paper_exps")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["fad", "ead", "self_bleu", "sbert", "vendi"],
        choices=["fad", "ead", "self_bleu", "sbert", "vendi"],
        help="Metrics to compute. fad=final-answer diversity over `pred`. self_bleu=1-mean Self-BLEU.",
    )
    parser.add_argument("--tokenizer", default="allenai/OLMo-3-1025-7B", help="HF tokenizer for EAD")
    parser.add_argument("--sbert-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="CSV output path (default: $WORK/bayesrl_paper_exps/output_diversity.csv)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute (model, step) rows that already exist in the CSV")
    args = parser.parse_args()
    main(args)
