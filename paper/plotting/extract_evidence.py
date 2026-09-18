"""Extract the manuscript's measured evidence from the repository JSON records.

Every number the paper reports comes through this script, so the figure inputs
and the table inputs share one verified path. Run from the paper directory:

    python3 plotting/extract_evidence.py

Writes CSVs into data/ and prints a verification report. Nothing here invents a
value: a missing source file is a hard error, not a filled-in default.
"""

import csv
import json
import os
import random
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)  # repository root; holds results/
RES = os.path.join(REPO, "results")
DATA = os.path.join(ROOT, "data")

# interference_chain history order (index -> semantic role)
ROLES = {
    0: "initial",
    1: "reference_1",
    2: "superseded",
    3: "reference_2",
    4: "latest",
}
ROLE_LABEL = {
    "initial": "Initial record",
    "reference_1": "Reference 1",
    "superseded": "Superseded update",
    "reference_2": "Reference 2",
    "latest": "Latest update",
}


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        raise SystemExit(f"MISSING SOURCE: {p}")
    with open(p) as fh:
        return json.load(fh)


def write_csv(name, rows, fields):
    os.makedirs(DATA, exist_ok=True)
    p = os.path.join(DATA, name)
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote data/{name}  ({len(rows)} rows)")


# --------------------------------------------------------------------------
# A. ten-prefix controlled intervention
# --------------------------------------------------------------------------
def extract_pilot10():
    d = load("tango_v4_interference_chain_kv_credit_pilot10.json")
    base = d["baseline_critical_accuracy"]
    rows = []
    for b in d["blocks"]:
        role = {"initial": "initial", "reference": "reference",
                "superseded_update": "superseded",
                "latest_update": "latest"}[b["role"]]
        rows.append({
            "index": b["index"],
            "role": role,
            "role_raw": b["role"],
            "patch_source": b["patch_source"],
            "baseline_accuracy": base,
            "patched_accuracy": b["critical_accuracy"],
            "accuracy_delta": b["critical_accuracy"] - base,
            "correct_score_effect": b["correct_score_effect"],
            "margin_effect": b["margin_effect"],
        })
    # disambiguate the two reference blocks
    seen = 0
    for r in rows:
        if r["role"] == "reference":
            seen += 1
            r["role"] = f"reference_{seen}"
    write_csv("pilot10.csv", rows, list(rows[0].keys()))
    return d, rows


# --------------------------------------------------------------------------
# B. 200-prefix signed credit
# --------------------------------------------------------------------------
def extract_credit200():
    d = load("tango_v2_teacher_credit200.json")
    recs = d["records"]
    rows = []
    for r in recs:
        rows.append({
            "prefix_id": r["prefix_id"],
            "history_index": r["history_index"],
            "role": ROLES[r["history_index"]],
            "memory_advantage": r["memory_advantage"],
            "Q_memory_full": r["Q_memory_full"],
            "donor_count": r["donor_count"],
        })
    write_csv("credit200.csv", rows, list(rows[0].keys()))

    # per-role summary with a bootstrap over prefixes. Each prefix contributes
    # exactly one record per role, so resampling records == resampling prefixes.
    random.seed(0)
    summary = []
    for i in range(5):
        role = ROLES[i]
        v = [r["memory_advantage"] for r in recs if r["history_index"] == i]
        bs = sorted(st.mean(random.choices(v, k=len(v))) for _ in range(4000))
        summary.append({
            "role": role,
            "label": ROLE_LABEL[role],
            "n": len(v),
            "mean": st.mean(v),
            "median": st.median(v),
            "sd": st.pstdev(v),
            "mean_abs": st.mean([abs(x) for x in v]),
            "pos_frac": sum(x > 0 for x in v) / len(v),
            "ci_lo": bs[100],
            "ci_hi": bs[-100],
        })
    write_csv("credit200_by_role.csv", summary, list(summary[0].keys()))

    # per-prefix joint pattern
    by = {}
    for r in recs:
        by.setdefault(r["prefix_id"], {})[r["history_index"]] = r["memory_advantage"]
    joint = [{"prefix_id": p, "superseded": v[2], "latest": v[4],
              "reference_max_abs": max(abs(v[1]), abs(v[3]))}
             for p, v in sorted(by.items())]
    write_csv("credit200_per_prefix.csv", joint, list(joint[0].keys()))

    n_both = sum(1 for r in joint if r["latest"] > 0 and r["superseded"] < 0)
    n_either = sum(1 for r in joint if r["latest"] > 0 or r["superseded"] < 0)

    def auroc(P, N):
        c = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in P for n in N)
        return c / (len(P) * len(N))

    sup = [r["superseded"] for r in joint]
    lat = [r["latest"] for r in joint]
    non = [r["reference_max_abs"] for r in joint]  # placeholder, recomputed below
    refs = [r["memory_advantage"] for r in recs if r["history_index"] in (0, 1, 3)]
    stats = {
        "prefixes": len(joint),
        "blocks": len(recs),
        "both_signs": n_both,
        "both_signs_frac": n_both / len(joint),
        "either_sign": n_either,
        "either_sign_frac": n_either / len(joint),
        "auroc_superseded_vs_rest": auroc(refs, sup),
        "auroc_latest_vs_rest": auroc(lat, refs),
    }
    with open(os.path.join(DATA, "credit200_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print("  wrote data/credit200_stats.json")
    return d, summary, joint, stats


# --------------------------------------------------------------------------
# C. MiniWoB paired online comparison
# --------------------------------------------------------------------------
# The source file names methods differently in `successes` / `per_task_successes`
# and in `mean_top1_margin`. This join is explicit so a rename cannot silently
# pair one method's success count with another's margin.
MINIWOB_KEYS = {
    "base": "base",
    "shared_warm_start": "warm",
    "return_weighted": "return",
    "successful_only_action_ce": "action_ce",
    "shuffled_return": "shuffle",
}
MINIWOB_LABEL = {
    "base": "Frozen base",
    "shared_warm_start": "Shared warm start",
    "return_weighted": "Return-weighted",
    "successful_only_action_ce": "Successful-only CE",
    "shuffled_return": "Shuffled return",
}


def extract_miniwob():
    d = load("tango_miniwob_click4_s701_800.json")
    rows = []
    for m, n in d["successes"].items():
        mk = MINIWOB_KEYS[m]
        fam = d["per_task_successes"][m]
        rows.append({
            "method": m,
            "label": MINIWOB_LABEL[m],
            "successes": n,
            "cases": d["paired_cases"],
            "success_rate": n / d["paired_cases"],
            "mean_top1_margin": d["mean_top1_margin"][mk],
            "click_color": fam["click-color"],
        })
    write_csv("miniwob.csv", rows, list(rows[0].keys()))
    return d, rows


# --------------------------------------------------------------------------
# D. return-conditioned KV precursor (margin up, ranking flat)
# --------------------------------------------------------------------------
def extract_gonogo():
    d = load("gonogo_current.json")
    rows = [
        {"method": "initial", "return_margin": d["heldout_return_margin"]["initial"],
         "top1": d["heldout_action_ranking"]["initial"]["top1"],
         "mrr": d["heldout_action_ranking"]["initial"]["mrr"]},
        {"method": "return_kv",
         "return_margin": d["heldout_return_margin"]["return_kv_epoch_20"],
         "top1": d["heldout_action_ranking"]["return_kv_epoch_20"]["top1"],
         "mrr": d["heldout_action_ranking"]["return_kv_epoch_20"]["mrr"]},
        {"method": "shuffled_return",
         "return_margin": d["heldout_return_margin"]["shuffled_return_epoch_20"],
         "top1": d["heldout_action_ranking"]["shuffled_return_epoch_20"]["top1"],
         "mrr": d["heldout_action_ranking"]["shuffled_return_epoch_20"]["mrr"]},
        {"method": "action_ce",
         "return_margin": d["heldout_return_margin"]["successful_action_ce_epoch_20"],
         "top1": d["heldout_action_ranking"]["successful_action_ce_epoch_20"]["top1"],
         "mrr": d["heldout_action_ranking"]["successful_action_ce_epoch_20"]["mrr"]},
    ]
    write_csv("gonogo.csv", rows, list(rows[0].keys()))
    return d, rows


# --------------------------------------------------------------------------
# E. transport energy / gate decay
# --------------------------------------------------------------------------
def extract_transport():
    g = load("gonogo_current.json")["low_energy_transport"]
    rows = []
    for k in ("unregularized", "lambda_3000", "lambda_10000", "lambda_100000"):
        rows.append({"setting": k,
                     "return_margin": g[k]["return_margin"],
                     "mean_transport_energy": g[k]["mean_transport_energy"]})
    write_csv("transport_energy.csv", rows, list(rows[0].keys()))

    t = load("tango_memory_transport_stage2_full.json")["memory_gate_summary"]
    grows = []
    for k, v in t.items():
        fam, kind, hist = k.split("|")
        grows.append({"family": fam, "kind": kind,
                      "history_index": int(hist.split("_")[1]),
                      "prefixes": v["prefixes"], "mean_gate": v["mean_gate"]})
    grows.sort(key=lambda r: (r["family"], r["kind"], r["history_index"]))
    write_csv("gate_decay.csv", grows, list(grows[0].keys()))

    ratio = g["unregularized"]["mean_transport_energy"] / g["lambda_3000"]["mean_transport_energy"]
    retained = g["lambda_3000"]["return_margin"] / g["unregularized"]["return_margin"]
    return g, ratio, retained


def main():
    print("Extracting evidence -> data/\n")
    print("A. ten-prefix controlled intervention")
    p10, brows = extract_pilot10()
    print("B. 200-prefix signed credit")
    c200, summary, joint, cstats = extract_credit200()
    print("C. MiniWoB paired online comparison")
    mw, mrows = extract_miniwob()
    print("D. return-conditioned KV precursor")
    gg, grows = extract_gonogo()
    print("E. transport energy and gate decay")
    tr, ratio, retained = extract_transport()

    print("\n" + "=" * 72)
    print("VERIFICATION REPORT")
    print("=" * 72)
    print(f"A  model={p10['model']}  target={p10['target']}  layers={p10['layers']}")
    print(f"A  baseline critical accuracy = {p10['baseline_critical_accuracy']}")
    for b in brows:
        print(f"     patch {b['role']:<13} -> acc {b['patched_accuracy']:.2f}"
              f"   score effect {b['correct_score_effect']:+.6f}")
    print(f"A  localization AUROC = "
          f"{p10['diagnostic_summary']['update_vs_reference_localization_auroc']}")
    print(f"A  sign accuracy = {p10['diagnostic_summary']['update_block_sign_accuracy']}")
    print()
    print(f"B  prefixes={cstats['prefixes']} blocks={cstats['blocks']}")
    print(f"   {'role':<13}{'mean':>11}{'pos frac':>10}{'bootstrap 95%':>24}")
    for s in summary:
        print(f"   {s['role']:<13}{s['mean']:>+11.6f}{s['pos_frac']:>10.1%}"
              f"   [{s['ci_lo']:+.6f}, {s['ci_hi']:+.6f}]")
    print(f"   per-prefix latest>0 AND superseded<0 : {cstats['both_signs']}"
          f"/{cstats['prefixes']} = {cstats['both_signs_frac']:.3f}")
    print(f"   per-prefix at least one sign         : {cstats['either_sign']}"
          f"/{cstats['prefixes']} = {cstats['either_sign_frac']:.3f}")
    print(f"   AUROC superseded vs non-update       : {cstats['auroc_superseded_vs_rest']:.3f}")
    print(f"   AUROC latest     vs non-update       : {cstats['auroc_latest_vs_rest']:.3f}")
    print()
    print(f"C  status={mw['status']} paired={mw['paired_cases']}")
    for r in mrows:
        print(f"   {r['label']:<22} {r['successes']:>4}/{r['cases']}"
              f"  margin {r['mean_top1_margin']:.4f}  click-color {r['click_color']}")
    print()
    print("D  return margin vs ranking (held-out)")
    for r in grows:
        print(f"   {r['method']:<18} margin {r['return_margin']:.4f}  top1 {r['top1']:.2f}")
    print()
    print(f"E  lambda=3000 energy ratio = {ratio:.2f}x  margin retained = {retained:.1%}")
    print("\nAll source files present; no value was defaulted.")


if __name__ == "__main__":
    main()
