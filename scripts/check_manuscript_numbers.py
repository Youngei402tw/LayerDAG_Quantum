"""Cross-check every decimal number in paper/qai_paper.tex against values derived from result JSON.

Run from the repository root:  python scripts/check_manuscript_numbers.py
Prints any 3-4 decimal number in the manuscript that is not derivable from the machine-readable results
(those must be reviewed by hand), and fails if a required headline value is missing from the manuscript.
"""
import json, re, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
tex = (ROOT / "paper/qai_paper.tex").read_text()
P = json.loads((ROOT / "experiments/qai_invariant_final_20261102/final_result.json").read_text())
T = json.loads((ROOT / "experiments/qai_transformer_control_20261106/control_result.json").read_text())
S = json.loads((ROOT / "experiments/schedule_dag_final_20261028/final_result.json").read_text())
R = json.loads((ROOT / "experiments/structured_transfer_final_20261102/final_result.json").read_text())
C = json.loads((ROOT / "experiments/critical_rewrite_utility_final_20261029/final_result.json").read_text())
W = json.loads((ROOT / "experiments/routed_rewrite_development_revision2_20261101/development_result.json").read_text())

vals = set()
def add(x, nd=(4,)):
    for n in nd:
        vals.add(f"{abs(x):.{n}f}".lstrip("0") or "0")
        vals.add(f"{abs(x):.{n}f}")

for m in ("dag", "grid"):
    for s, r in P["aggregate"][m].items():
        for k in ("auc_mean", "auc_sample_std", "f1_mean", "accuracy_mean"):
            add(r[k])
for s, r in T["aggregate"].items():
    for k in ("auc_mean", "auc_sample_std", "f1_mean"):
        add(r[k])
d = P["far_width_auc_paired_differences"]
for x in d: add(x)
add(statistics.mean(d)); add(statistics.stdev(d)); add(P["far_width_f1_mean_difference"])
t = 2.7764451051977987 * statistics.stdev(d) / 5 ** 0.5
add(statistics.mean(d) - t); add(statistics.mean(d) + t)
A = P["aggregate"]
for s in ("same", "near_width"):
    add(A["dag"][s]["auc_mean"] - A["grid"][s]["auc_mean"])
add(A["grid"]["same"]["auc_mean"] - A["grid"]["renamed"]["auc_mean"])
for s in P["seed_results"][0]["models"]["dag"]:
    if s != "training":
        add(P["seed_results"][0]["models"]["dag"][s]["positive_fraction"], (3,))
gk = "canonical_grid"
for s in R["aggregate"]["dag"]:
    add(R["aggregate"]["dag"][s]["auc_mean"]); add(R["aggregate"][gk][s]["auc_mean"])
    add(statistics.mean(R["dag_minus_grid_auc"][s]))
for m in ("dag", "canonical_grid"):
    for s in ("canonical_test", "larger_test"):
        add(S["aggregate"][m][s]["auc_mean"]); add(S["aggregate"][m][s]["f1_mean"])
for x in S["larger_auc_paired_differences"]: add(x)
add(float(C["recovery_mean_difference"]))
add(W["methods"]["dag"]["evaluation"]["mean_cx_reduction"], (2, 4))
add(W["methods"]["canonical_grid"]["evaluation"]["mean_cx_reduction"], (2, 4))
# protocol thresholds / constants that legitimately appear
vals |= {"0.03", ".03", "0.05", ".05", "0.99", "0.0", "1.0", "0.5"}

found = re.findall(r"(?<![\w.])[+-]?(?:0)?\.\d{2,4}(?!\d)", tex)
unknown = sorted({f.lstrip("+-") for f in found} - vals)
print("manuscript decimals checked:", len(set(found)))
print("NOT derivable from results (review by hand):", unknown if unknown else "none")
required = ["0.9252", "0.8807", "0.7471", "0.0444", "0.1097", "+0.0400", "+0.0488", "80,833", "84,961", "88,065", "0.0514", "0.0025"]
missing = [r for r in required if r not in tex]
print("required headline values missing:", missing if missing else "none")
sys.exit(1 if (missing or unknown) else 0)
