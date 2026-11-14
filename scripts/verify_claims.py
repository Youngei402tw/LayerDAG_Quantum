"""Verify every numeric claim used in the QAI manuscript against machine-readable results.

Run from the repository root:  python scripts/verify_claims.py
Exits non-zero on any mismatch. Read-only: never modifies result files.
"""
import hashlib, json, math, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAIL = []

def load(p):
    return json.loads((ROOT / p).read_text())

def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()

def check(name, observed, expected, tol):
    ok = abs(observed - expected) <= tol
    print(f"[{'OK ' if ok else 'BAD'}] {name}: observed={observed:.8f} expected={expected} tol={tol}")
    if not ok:
        FAIL.append(name)

# ---------------- hashes ----------------
EXPECTED_HASHES = {
    "experiments/qai_invariant_final_20261102/final_result.json": "23f47f27337d0cfc56242926dd3988db34342cf5ea26c5d871e822d9cdb21247",
    "experiments/qai_transformer_control_20261106/control_result.json": "6989761a20ef5b3b0c7ea1f50b5f0cee8aced285fec62ac3d7b85fa3747f9cb0",
    "experiments/schedule_dag_final_20261028/final_result.json": "9144132e2bcdb0aa0628cb62f75dc017d28d52498366aca178b2e5569625b435",
    "experiments/structured_transfer_final_20261102/final_result.json": "5235c71446f61e66251aea5e3482d39921f7cf42b4b93f40eb611081bb255692",
    "experiments/qai_algorithm_suite_20261114/data_manifest.json": "9e161851cbace5251544aa22ec45635c3a38d9425c9f55a31d0bf4ce2a0980b5",
    "experiments/qai_algorithm_suite_20261114/phaseA_result.json": "5ffa0bf4835df0cef514183329a2a204003039cfe185647b2918201abdffac97",
    "experiments/qai_algorithm_suite_20261114/phaseB_result.json": "4ae4d095d4aef94698707d264bc913db4983c6083b0c03613bb4bca7b3895cdf",
    "experiments/qai_augmentation_control_20261113/augmentation_result.json": "709b55490388103a7bcc48b02d4c301a04ed4521eb85279fed3b0a77f721efd2",
    "experiments/qai_connectivity_controls_20261113/controls_result.json": "52f063262faf8e938df42ca44d1c5fadf4db2d1f6c866533819212764c4fc7d3",
}
for p, h in EXPECTED_HASHES.items():
    ok = sha(p) == h
    print(f"[{'OK ' if ok else 'BAD'}] sha256 {p}")
    if not ok:
        FAIL.append("hash " + p)

# ---------------- primary frozen final ----------------
P = load("experiments/qai_invariant_final_20261102/final_result.json")
A = P["aggregate"]
assert P["seeds"] == [111, 112, 113, 114, 115]
assert P["updates_per_model"] == 4000
expected_primary = {
    ("dag", "same"): (0.9722, 0.0009, 0.9383), ("grid", "same"): (0.9391, 0.0028, 0.8919),
    ("dag", "near_width"): (0.9362, 0.0008, 0.8749), ("grid", "near_width"): (0.8853, 0.0027, 0.8034),
    ("dag", "far_width"): (0.9252, 0.0020, 0.8458), ("grid", "far_width"): (0.8807, 0.0045, 0.8011),
    ("dag", "renamed"): (0.9722, 0.0009, 0.9383), ("grid", "renamed"): (0.8294, 0.0053, 0.8199),
}
for (m, s), (auc, sd, f1) in expected_primary.items():
    check(f"primary {m} {s} AUROC", A[m][s]["auc_mean"], auc, 5e-5)
    check(f"primary {m} {s} AUROC sd", A[m][s]["auc_sample_std"], sd, 5e-5)
    check(f"primary {m} {s} F1", A[m][s]["f1_mean"], f1, 5e-5)

diffs = P["far_width_auc_paired_differences"]
for got, exp in zip(diffs, [0.04186045, 0.04192684, 0.04169740, 0.04823048, 0.04835311]):
    check("paired far AUROC diff", got, exp, 5e-9)
# recompute paired differences from per-seed values
for i, sr in enumerate(P["seed_results"]):
    d = sr["models"]["dag"]["far_width"]["auc"] - sr["models"]["grid"]["far_width"]["auc"]
    check(f"recomputed paired diff seed {sr['seed']}", d, diffs[i], 1e-12)
mean = statistics.mean(diffs); sd = statistics.stdev(diffs)
check("mean paired far AUROC diff", mean, 0.04441366, 5e-9)
t975_df4 = 2.7764451051977987
lo, hi = mean - t975_df4 * sd / math.sqrt(5), mean + t975_df4 * sd / math.sqrt(5)
check("t-interval low", lo, 0.0400, 5e-5); check("t-interval high", hi, 0.0488, 5e-5)
print(f"      seed-level paired sd = {sd:.6f}; 95% t interval = [{lo:.5f}, {hi:.5f}]")
check("far F1 diff", P["far_width_f1_mean_difference"], 0.04460911, 5e-9)
check("grid relabel AUROC drop", A["grid"]["same"]["auc_mean"] - A["grid"]["renamed"]["auc_mean"], 0.1097, 5e-5)
check("DAG max relabel logit diff", P["dag_max_qubit_relabel_logit_difference"], 0.0, 0.0)
check("DAG same-vs-relabel metric diff", P["dag_max_same_vs_relabel_metric_difference"], 0.0, 0.0)
# gate conditions (as frozen)
gate = [all(d > 0 for d in diffs), mean >= 0.03,
        A["dag"]["far_width"]["f1_mean"] > A["grid"]["far_width"]["f1_mean"],
        P["dag_max_qubit_relabel_logit_difference"] <= 1e-6,
        P["dag_max_same_vs_relabel_metric_difference"] <= 1e-6]
print("      frozen gate conditions:", gate, "recorded positive_claim_passed =", P["positive_claim_passed"])
if not (all(gate) and P["positive_claim_passed"]):
    FAIL.append("gate")
params = {m: {sr["models"][m]["training"]["parameters"] for sr in P["seed_results"]} for m in ("dag", "grid")}
print("      parameters:", params)
if params != {"dag": {80833}, "grid": {84961}}:
    FAIL.append("params")

# ---------------- Transformer control ----------------
T = load("experiments/qai_transformer_control_20261106/control_result.json")
assert T["seeds"] == P["seeds"] and T["status"] == "descriptive_control_not_part_of_confirmatory_gate"
for s, (auc, sd) in {"same": (0.8888, 0.0010), "near_width": (0.7467, 0.0070),
                     "far_width": (0.7471, 0.0128), "renamed": (0.7806, 0.0064)}.items():
    check(f"transformer {s} AUROC", T["aggregate"][s]["auc_mean"], auc, 5e-5)
    check(f"transformer {s} AUROC sd", T["aggregate"][s]["auc_sample_std"], sd, 5e-5)
tp = {sr["training"]["parameters"] for sr in T["seed_results"]}
print("      transformer parameters:", tp)
if tp != {88065}:
    FAIL.append("transformer params")

# ---------------- connectivity controls (post-confirmatory, 2026-11-13) ----------------
K = load("experiments/qai_connectivity_controls_20261113/controls_result.json")
assert K["seeds"] == P["seeds"] and K["dataset_hashes"] == P["dataset_hashes"]
KA = K["aggregate"]
expected_controls = {
    "pointer_grid": {"same": (0.9895, 0.0010), "near_width": (0.9447, 0.0026), "far_width": (0.9274, 0.0050), "renamed": (0.9279, 0.0044)},
    "pointer_transformer": {"same": (0.9481, 0.0019), "near_width": (0.7533, 0.0168), "far_width": (0.6900, 0.0192), "renamed": (0.8838, 0.0035)},
    "role_ablated_dag": {"same": (0.9670, 0.0013), "near_width": (0.9282, 0.0019), "far_width": (0.9199, 0.0025), "renamed": (0.9670, 0.0013)},
}
rec = {sr["seed"]: sr["models"] for sr in P["seed_results"]}
for m, splits in expected_controls.items():
    for s, (auc, sd) in splits.items():
        check(f"control {m} {s} AUROC", KA[m][s]["auc_mean"], auc, 5e-5)
        check(f"control {m} {s} AUROC sd", KA[m][s]["auc_sample_std"], sd, 5e-5)
    d = [rec[r["seed"]]["dag"]["far_width"]["auc"] - r["far_width"]["auc"] for r in K["seed_results"][m]]
    for got, stored in zip(d, K["comparisons"][m]["recorded_dag_minus_model_far_auc"]):
        check(f"control {m} recomputed paired diff", got, stored, 1e-12)
dp = K["comparisons"]["pointer_grid"]["recorded_dag_minus_model_far_auc"]
mp, sp = statistics.mean(dp), statistics.stdev(dp); hp = t975_df4 * sp / math.sqrt(5)
check("pointer GRU paired mean", mp, -0.0023, 5e-5); check("pointer GRU paired sd", sp, 0.0049, 5e-5)
check("pointer GRU interval low", mp - hp, -0.0083, 5e-5); check("pointer GRU interval high", mp + hp, 0.0038, 5e-5)
check("pointer GRU DAG wins", sum(x > 0 for x in dp), 2, 0)
check("pointer GRU minus grid far", statistics.mean(K["comparisons"]["pointer_grid"]["model_minus_recorded_grid_far_auc"]), 0.0467, 5e-5)
check("pointer GRU relabel drop", KA["pointer_grid"]["same"]["auc_mean"] - KA["pointer_grid"]["renamed"]["auc_mean"], 0.0615, 5e-5)
check("pointer GRU far F1", KA["pointer_grid"]["far_width"]["f1_mean"], 0.8081, 5e-5)
f1d = [rec[r["seed"]]["dag"]["far_width"]["f1"] - r["far_width"]["f1"] for r in K["seed_results"]["pointer_grid"]]
check("DAG beats pointer GRU far F1 in all seeds", sum(x > 0 for x in f1d), 5, 0)
check("pointer transformer paired mean", K["comparisons"]["pointer_transformer"]["mean"], 0.2352, 5e-5)
da = K["comparisons"]["role_ablated_dag"]["recorded_dag_minus_model_far_auc"]
ma, ha = statistics.mean(da), t975_df4 * statistics.stdev(da) / math.sqrt(5)
check("role ablation mean", ma, 0.0053, 5e-5); check("role ablation low", ma - ha, 0.0019, 5e-5); check("role ablation high", ma + ha, 0.0086, 5e-5)
check("role ablation DAG wins", sum(x > 0 for x in da), 5, 0)
check("role-ablated DAG max relabel logit diff", max(r["max_relabel_logit_difference_first64"] for r in K["seed_results"]["role_ablated_dag"]), 0.0, 0.0)
kp = {m: {r["training"]["parameters"] for r in v} for m, v in K["seed_results"].items()}
print("      control parameters:", kp, "readings:", K["readings"])
if kp != {"pointer_grid": {87457}, "pointer_transformer": {89729}, "role_ablated_dag": {80833}}:
    FAIL.append("control params")
if K["readings"] != {"pointer_grid_H1": "information_explains_gap", "pointer_transformer_descriptive": "topology_supported", "role_ablated_dag_descriptive": "roles_not_needed"}:
    FAIL.append("control readings")
E = load("experiments/qai_env_check_seed112_20261113/spot_reproduction.json")
check("env check max AUROC deviation", max(v["abs_diff"] for m in E["models"].values() for v in m.values() if isinstance(v, dict)), 0.0, 0.0)

# ---------------- relabeling-augmentation control (post-confirmatory, 2026-11-13) ----------------
G = load("experiments/qai_augmentation_control_20261113/augmentation_result.json")
assert G["seeds"] == P["seeds"] and G["dataset_hashes"] == P["dataset_hashes"]
GA = G["aggregate"]["aug_pointer_grid"]
for s_, (auc, sd) in {"same": (0.9757, 0.0018), "near_width": (0.9296, 0.0012), "far_width": (0.9149, 0.0027),
                      "renamed": (0.9721, 0.0017), "far_renamed": (0.9173, 0.0029)}.items():
    check(f"aug pointer {s_} AUROC", GA[s_]["auc_mean"], auc, 5e-5); check(f"aug pointer {s_} AUROC sd", GA[s_]["auc_sample_std"], sd, 5e-5)
check("aug pointer same gap", GA["same_relabel_gap_mean"], 0.0035, 5e-5)
check("aug pointer far gap", GA["far_relabel_gap_mean"], -0.0024, 5e-5)
check("aug pointer mean |dp| same", GA["same_pair_changes"]["mean_abs_probability_change"], 0.053, 5e-4)
check("aug pointer max logit same", GA["same_pair_changes"]["max_logit_change"], 10.8, 0.05)
dg = [rec[r["seed"]]["dag"]["far_width"]["auc"] - r["far_width"]["auc"] for r in G["seed_results"]["aug_pointer_grid"]]
mg, hg = statistics.mean(dg), t975_df4 * statistics.stdev(dg) / math.sqrt(5)
check("aug pointer DAG lead", mg, 0.0103, 5e-5); check("aug pointer lead low", mg - hg, 0.0055, 5e-5); check("aug pointer lead high", mg + hg, 0.0150, 5e-5)
check("aug pointer DAG wins", sum(x > 0 for x in dg), 5, 0)
check("aug grid far AUROC", G["aggregate"]["aug_grid"]["far_width"]["auc_mean"], 0.8657, 5e-5)
check("pointer GRU max logit (control)", max(r["max_relabel_logit_difference_first64"] for r in K["seed_results"]["pointer_grid"]), 19.2, 0.05)
check("pointer Transformer max logit (control)", max(r["max_relabel_logit_difference_first64"] for r in K["seed_results"]["pointer_transformer"]), 27.2, 0.05)
if G["reading_aug_pointer_grid"] != "intermediate":
    FAIL.append("augmentation reading")

# ---------------- standard-algorithm suite (post-confirmatory, 2026-11-14) ----------------
SUITE = "experiments/qai_algorithm_suite_20261114"
MAN = load(f"{SUITE}/data_manifest.json")
check("suite training circuits", MAN["statistics"]["training"]["circuits"], 1100, 0); check("suite validation circuits", MAN["statistics"]["validation"]["circuits"], 264, 0)
for s_ in ("same", "near_width", "far_width"): check(f"suite {s_} circuits", MAN["statistics"][s_]["circuits"], 528, 0)
check("suite far max gates", MAN["statistics"]["far_width"]["gates_max"], 1191, 0)
check("suite label disagreement training", 1 - MAN["statistics"]["training"]["node_label_agreement_A_B"], 0.28, 0.005)
check("suite label disagreement far", 1 - MAN["statistics"]["far_width"]["node_label_agreement_A_B"], 0.36, 0.005)
fg = [v["gates_mean"] for v in MAN["statistics"]["far_width"]["families"].values()]
check("suite far family size ratio", max(fg) / min(fg), 33, 0.5); check("suite families", len(MAN["families"]), 11, 0)
if not all(v["labels_recomputed_and_equal"] for v in MAN["renaming_checks"].values()): FAIL.append("suite renaming label check")
EXPECT = {
    "A": {"readings": {"R1_connectivity": "replicated", "R2_dag_vs_pointer": "tie", "R3_augmentation": "suffices", "R4_dag_invariance": "exact_within_1e-6"},
          "macro": {("dag", "same"): (0.9803, 0.0012), ("pointer_grid", "same"): (0.9920, 0.0004), ("aug_pointer_grid", "same"): (0.9884, 0.0004), ("grid", "same"): (0.9663, 0.0009),
                    ("dag", "far_width"): (0.9455, 0.0010), ("pointer_grid", "far_width"): (0.9442, 0.0040), ("aug_pointer_grid", "far_width"): (0.9450, 0.0044), ("grid", "far_width"): (0.8617, 0.0231),
                    ("dag", "far_relabeled"): (0.9455, 0.0010), ("pointer_grid", "far_relabeled"): (0.9205, 0.0056), ("aug_pointer_grid", "far_relabeled"): (0.9463, 0.0039), ("grid", "far_relabeled"): (0.7961, 0.0234)},
          "diff": {"grid": (0.0838, 5), "pointer_grid": (0.0013, 3), "aug_pointer_grid": (0.0005, 4)}},
    "B": {"readings": {"R1_connectivity": "replicated", "R2_dag_vs_pointer": "graph_advantage", "R3_augmentation": "insufficient", "R4_dag_invariance": "exact_within_1e-6", "roles": "roles_contribute"},
          "macro": {("dag", "same"): (0.9635, 0.0016), ("pointer_grid", "same"): (0.9562, 0.0017), ("aug_pointer_grid", "same"): (0.9476, 0.0017), ("grid", "same"): (0.9222, 0.0009), ("role_ablated_dag", "same"): (0.9376, 0.0013),
                    ("dag", "far_width"): (0.9354, 0.0006), ("pointer_grid", "far_width"): (0.8991, 0.0046), ("aug_pointer_grid", "far_width"): (0.8933, 0.0055), ("grid", "far_width"): (0.8243, 0.0105), ("role_ablated_dag", "far_width"): (0.9002, 0.0026),
                    ("dag", "far_relabeled"): (0.9354, 0.0006), ("pointer_grid", "far_relabeled"): (0.8516, 0.0030), ("aug_pointer_grid", "far_relabeled"): (0.8921, 0.0039), ("grid", "far_relabeled"): (0.7505, 0.0079), ("role_ablated_dag", "far_relabeled"): (0.9002, 0.0026)},
          "diff": {"grid": (0.1111, 5), "pointer_grid": (0.0363, 5), "aug_pointer_grid": (0.0421, 5), "role_ablated_dag": (0.0352, 5)}},
}
SUITE_RESULT = {}
for phase, exp in EXPECT.items():
    Q = load(f"{SUITE}/phase{phase}_result.json"); SUITE_RESULT[phase] = Q; QA = Q["aggregate"]
    assert Q["seeds"] == P["seeds"] and Q["data_hashes"] == MAN["hashes"][phase]
    units = {k: [load(f"{SUITE}/phase{phase}_{k}_seed{sd_}.json") for sd_ in Q["seeds"]] for k in QA}
    for k, rows_ in units.items():  # every unit carries the manifest check, the right parameter count and 4,000 updates
        if not all(u["data_hashes_match_manifest"] and u["training"]["updates"] == 4000 and u["training"]["parameters"] == QA[k]["parameters"] for u in rows_): FAIL.append(f"suite unit {phase} {k}")
    for (k, s_), (m_, sd_) in exp["macro"].items():
        vals_ = [u["splits"][s_]["macro_auc"] for u in units[k]]
        check(f"suite {phase} {k} {s_} macro AUROC (recomputed from units)", statistics.mean(vals_), m_, 5e-5)
        check(f"suite {phase} {k} {s_} macro AUROC sd", statistics.stdev(vals_), sd_, 5e-5)
        check(f"suite {phase} {k} {s_} aggregate equals units", QA[k]["splits"][s_]["macro_auc_mean"], statistics.mean(vals_), 1e-12)
    for k, (m_, wins_) in exp["diff"].items():
        d_ = [a_["splits"]["far_width"]["macro_auc"] - b_["splits"]["far_width"]["macro_auc"] for a_, b_ in zip(units["dag"], units[k])]
        check(f"suite {phase} DAG minus {k} far (recomputed)", statistics.mean(d_), m_, 5e-5); check(f"suite {phase} DAG wins vs {k}", sum(x > 0 for x in d_), wins_, 0)
        check(f"suite {phase} stored comparison {k}", Q["comparisons_far_macro_auc"][f"dag_minus_{k}"]["mean"], statistics.mean(d_), 1e-12)
    for key, value in exp["readings"].items():
        if Q["readings"][key] != value: FAIL.append(f"suite {phase} reading {key}")
    check(f"suite {phase} DAG max logit change", Q["readings"]["R4_dag_max_logit_change"], 0.0, 0.0)
    check(f"suite {phase} renamed pairs per seed", sum(v["pairs"] for v in units["dag"][0]["pair_changes"].values()), 2112, 0)
    print(f"      suite phase {phase} readings:", Q["readings"])
CA = SUITE_RESULT["A"]["comparisons_far_macro_auc"]; CB = SUITE_RESULT["B"]["comparisons_far_macro_auc"]; AA = SUITE_RESULT["A"]["aggregate"]; AB = SUITE_RESULT["B"]["aggregate"]
check("suite A pointer minus grid", CA["pointer_grid_minus_grid"]["mean"], 0.0825, 5e-5); check("suite A pointer wins vs grid", CA["pointer_grid_minus_grid"]["first_wins"], 5, 0)
check("suite A tie low", CA["dag_minus_pointer_grid"]["t95_low"], -0.0033, 5e-5); check("suite A tie high", CA["dag_minus_pointer_grid"]["t95_high"], 0.0060, 5e-5)
check("suite A pointer far relabel gap", AA["pointer_grid"]["gaps"]["far_relabeled"]["mean"], 0.0237, 5e-5); check("suite A pointer far reversal gap", AA["pointer_grid"]["gaps"]["far_reversed"]["mean"], 0.0362, 5e-5)
check("suite A grid far relabel gap", AA["grid"]["gaps"]["far_relabeled"]["mean"], 0.0656, 5e-5); check("suite A grid far reversal gap", AA["grid"]["gaps"]["far_reversed"]["mean"], 0.0828, 5e-5)
check("suite A aug same gap", AA["aug_pointer_grid"]["gaps"]["same_relabeled"]["mean"], 0.0005, 5e-5); check("suite A aug far gap", AA["aug_pointer_grid"]["gaps"]["far_relabeled"]["mean"], -0.0013, 5e-5)
check("suite A aug far mean |dp|", AA["aug_pointer_grid"]["pair_changes"]["far_relabeled"]["mean_abs_probability_change"], 0.072, 5e-4)
for k, f1 in (("dag", 0.9738), ("pointer_grid", 0.8112), ("aug_pointer_grid", 0.9004)): check(f"suite A {k} far F1", AA[k]["splits"]["far_width"]["f1_mean"], f1, 5e-5)
fam = {k: AA[k]["splits"]["far_width"]["family_auc_mean"] for k in AA}
for f_ in ("graphstate", "iqp", "qaoa"):
    if not min(fam["pointer_grid"][f_], fam["aug_pointer_grid"][f_]) > fam["dag"][f_]: FAIL.append(f"suite A family claim {f_}")
for f_ in ("qft", "qpe", "qv", "vqe"):
    if not fam["dag"][f_] > max(fam["pointer_grid"][f_], fam["aug_pointer_grid"][f_]): FAIL.append(f"suite A family claim {f_}")
check("suite B pointer minus grid", CB["pointer_grid_minus_grid"]["mean"], 0.0748, 5e-5)
check("suite B graph advantage low", CB["dag_minus_pointer_grid"]["t95_low"], 0.0303, 5e-5); check("suite B graph advantage high", CB["dag_minus_pointer_grid"]["t95_high"], 0.0422, 5e-5)
check("suite B aug lead low", CB["dag_minus_aug_pointer_grid"]["t95_low"], 0.0351, 5e-5); check("suite B aug lead high", CB["dag_minus_aug_pointer_grid"]["t95_high"], 0.0491, 5e-5)
check("suite B roles low", CB["dag_minus_role_ablated_dag"]["t95_low"], 0.0323, 5e-5); check("suite B roles high", CB["dag_minus_role_ablated_dag"]["t95_high"], 0.0381, 5e-5)
check("suite B roles minimum seed", min(CB["dag_minus_role_ablated_dag"]["differences"]), 0.033, 5e-4)
check("suite B aug max |relabel gap|", max(abs(AB["aug_pointer_grid"]["gaps"][v]["mean"]) for v in ("same_relabeled", "far_relabeled")), 0.0012, 5e-5)
if AB["dag"]["splits"]["far_width"]["macro_excluded_single_class"] != ["oracle"] or AA["dag"]["splits"]["far_width"]["macro_excluded_single_class"]: FAIL.append("suite excluded families")

# ---------------- earlier random-circuit study ----------------
S = load("experiments/schedule_dag_final_20261028/final_result.json")
SA = S["aggregate"]
check("schedule DAG same AUROC", SA["dag"]["canonical_test"]["auc_mean"], 0.8556, 5e-5)
check("schedule grid same AUROC", SA["canonical_grid"]["canonical_test"]["auc_mean"], 0.9406, 5e-5)
check("schedule DAG larger AUROC", SA["dag"]["larger_test"]["auc_mean"], 0.8078, 5e-5)
check("schedule grid larger AUROC", SA["canonical_grid"]["larger_test"]["auc_mean"], 0.7861, 5e-5)
check("schedule DAG larger F1", SA["dag"]["larger_test"]["f1_mean"], 0.6341, 5e-5)
check("schedule grid larger F1", SA["canonical_grid"]["larger_test"]["f1_mean"], 0.6055, 5e-5)
for got, exp in zip(S["larger_auc_paired_differences"], [0.0202, 0.0177, 0.0274]):
    check("schedule paired larger diff", got, exp, 5e-5)

# ---------------- structured-family study ----------------
R = load("experiments/structured_transfer_final_20261102/final_result.json")
RA = R["aggregate"]
gk = [k for k in RA if k != "dag"][0]
check("structured DAG both AUROC", RA["dag"]["both_shift"]["auc_mean"], 0.8902, 5e-5)
check("structured grid both AUROC", RA[gk]["both_shift"]["auc_mean"], 0.8495, 5e-5)
check("structured both diff", R["both_shift_auc_mean_difference"], 0.0407, 5e-5)
check("structured DAG both F1", RA["dag"]["both_shift"]["f1_mean"], 0.8351, 5e-5)
check("structured grid both F1", RA[gk]["both_shift"]["f1_mean"], 0.7928, 5e-5)
for got, exp in zip(R["dag_minus_grid_auc"]["both_shift"], [0.0321, 0.0243, 0.0533, 0.0414, 0.0525]):
    check("structured per-seed both diff", got, exp, 5e-5)
check("structured width-only diff", statistics.mean(R["dag_minus_grid_auc"]["width_shift"]), 0.0514, 5e-5)
check("structured depth-only diff", statistics.mean(R["dag_minus_grid_auc"]["depth_shift"]), 0.0025, 5e-5)
check("structured chemistry grid-minus-dag", -statistics.mean(R["dag_minus_grid_auc"]["heldout_chemistry"]), 0.0266, 5e-5)
for s in ("same_size", "width_shift", "depth_shift", "both_shift", "heldout_chemistry"):
    print(f"      structured {s}: DAG {RA['dag'][s]['auc_mean']:.4f}  grid {RA[gk][s]['auc_mean']:.4f}")

# ---------------- compute ledger ----------------
L = load("experiments/budget_ledger.json")
print("      budget ledger keys:", list(L)[:10])

print("\nFAILURES:", FAIL if FAIL else "none")
sys.exit(1 if FAIL else 0)
