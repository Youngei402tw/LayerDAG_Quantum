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
K = json.loads((ROOT / "experiments/qai_connectivity_controls_20261113/controls_result.json").read_text())
G = json.loads((ROOT / "experiments/qai_augmentation_control_20261113/augmentation_result.json").read_text())
SUITE = {ph: json.loads((ROOT / f"experiments/qai_algorithm_suite_20261114/phase{ph}_result.json").read_text()) for ph in ("A", "B")}
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
for m, agg in K["aggregate"].items():
    for sp, r in agg.items():
        for k in ("auc_mean", "auc_sample_std", "f1_mean"):
            add(r[k])
    add(agg["same"]["auc_mean"] - agg["renamed"]["auc_mean"])
for m, c in K["comparisons"].items():
    add(c["sample_std"]); add(c["mean"]); add(c["mean"] - c["t95_half_width"]); add(c["mean"] + c["t95_half_width"])
    add(statistics.mean(c["model_minus_recorded_grid_far_auc"]))
for m, agg in G["aggregate"].items():
    for sp in ("same", "near_width", "far_width", "renamed", "far_renamed"):
        for k in ("auc_mean", "auc_sample_std", "f1_mean"):
            add(agg[sp][k])
    add(agg["same_relabel_gap_mean"]); add(agg["far_relabel_gap_mean"])
    add(agg["same_pair_changes"]["mean_abs_probability_change"], (3,))
for m, c in G["comparisons"].items():
    add(c["mean"]); add(c["mean"] - c["t95_half_width"]); add(c["mean"] + c["t95_half_width"])
for ph, Q in SUITE.items():
    for kind, agg in Q["aggregate"].items():
        for sp, r in agg["splits"].items():
            for k in ("macro_auc_mean", "macro_auc_sample_std", "auc_mean", "f1_mean"):
                add(r[k])
        for g in agg["gaps"].values(): add(g["mean"])
        for c in agg["pair_changes"].values(): add(c["mean_abs_probability_change"], (3,))
    for c in Q["comparisons_far_macro_auc"].values():
        add(c["mean"]); add(c["sample_std"]); add(c["t95_low"]); add(c["t95_high"])
# protocol thresholds / constants that legitimately appear
vals |= {"0.03", ".03", "0.05", ".05", "0.99", "0.0", "1.0", "0.5", "0.02", "0.01", "0.005"}

found = re.findall(r"(?<![\w.])[+-]?(?:0)?\.\d{2,4}(?!\d)", tex)
unknown = sorted({f.lstrip("+-") for f in found} - vals)
print("manuscript decimals checked:", len(set(found)))
print("NOT derivable from results (review by hand):", unknown if unknown else "none")
required = ["0.9252", "0.8807", "0.7471", "0.0444", "0.1097", "+0.0400", "+0.0488", "80,833", "84,961", "88,065", "0.0514", "0.0025", ".9274", "-0.0023", "-0.0083", "+0.0038", "0.0615", "0.0053", "87,457", "89,729", "0.0103", "0.9149", "0.053",
            ".9455", ".9354", "+0.0838", "+0.0825", "+0.0013", "+0.0005", "0.0363", "0.0421", "0.0352", "+0.0748", "0.0237", "0.0362", "0.072", "1,100", "1,191", "2,112"]
missing = [r for r in required if r not in tex]
print("required headline values missing:", missing if missing else "none")
sys.exit(1 if (missing or unknown) else 0)
