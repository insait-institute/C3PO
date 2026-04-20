import pickle
from pathlib import Path

base_dir = Path(__file__).parents[0]

base_run = base_dir / "eval_preds/qwm7b_nmtron_ivon/eval_correctness.pkl"

with open(base_run, "rb") as f:
    base_run = pickle.load(f)

for run in ["qwm_nmtron_ivon_LR2.5_GS4_ESS1e9_MC4", "qwm_nmtron_ivon_LR2.5_GS16_ESS1e9", "qwm_nmtron_adamw_LR2.5_GS16"]:
    p, e, s, o = [], [], [], []
    with open(base_dir / "eval_preds" / run / "eval_correctness.pkl", "rb") as f:
        current_run = pickle.load(f)
    for uid, is_solved in current_run.items():
        if is_solved and base_run[uid]:
            p.append(uid)
        elif is_solved and not base_run[uid]:
            e.append(uid)
        elif not is_solved and base_run[uid]:
            s.append(uid)
        elif not is_solved and not base_run[uid]:
            o.append(uid)
    P, E, S, O = len([x for x in p if "aime" in x]), len([x for x in e if "aime" in x]), len([x for x in s if "aime" in x]), len([x for x in o if "aime" in x])
    print(f"Run: {run}")
    srr = P / (P + S)
    ndr = E / (P + E)
    sds = 2 * srr * ndr / (srr + ndr)
    nscr = (E - S) / (P + E + S)
    print(f"SRR: {srr:.3f}, NDR: {ndr:.3f}, SDS: {sds:.3f}, NSCR: {nscr:.3f}")
