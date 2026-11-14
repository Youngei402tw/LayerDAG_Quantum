"""Run code for the QAI algorithm-suite study.

Protocol: experiments/QAI_ALGORITHM_SUITE_PROTOCOL_20261114.md (frozen and pushed before this file was written).
Model classes and the training loop are those of the frozen studies; nothing here is tuned.

    python -m qdag.algorithm_suite_run prepare   --output DIR --cache FILE
    python -m qdag.algorithm_suite_run unit      --output DIR --cache FILE --phase A --kind dag --seed 111
    python -m qdag.algorithm_suite_run aggregate --output DIR --phase A
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata
from torch import nn
from torch.nn import functional as F

from .algorithm_suite import FAMILIES, commutation_labels, generate_suite
from .connectivity_controls import PointerGrid, RoleAblatedDag
from .core import Op
from .invariant_benchmark import InvariantDag, InvariantGrid, evaluate
from .schedule_benchmark import CircuitExample, critical_labels, examples_hash

SEEDS = (111, 112, 113, 114, 115)
SPLIT_SPECS = (("training", 15100, 1100, (4, 7)), ("validation", 15101, 264, (4, 7)), ("same", 15102, 528, (4, 7)),
               ("near_width", 15103, 528, (8, 12)), ("far_width", 15104, 528, (13, 20)))
RELABEL_SEEDS = {"same": 15105, "far_width": 15106}
VARIANTS = {"same_relabeled": "same", "far_relabeled": "far_width", "same_reversed": "same", "far_reversed": "far_width"}
TEST = ("same", "near_width", "far_width") + tuple(VARIANTS)
MODELS = {"dag": (InvariantDag, False), "grid": (InvariantGrid, False), "pointer_grid": (PointerGrid, False),
          "aug_pointer_grid": (PointerGrid, True), "role_ablated_dag": (RoleAblatedDag, False)}
EXPECTED_PARAMETERS = {"dag": 80833, "grid": 84961, "pointer_grid": 87457, "aug_pointer_grid": 87457, "role_ablated_dag": 80833}
PHASE_KINDS = {"A": ("dag", "grid", "pointer_grid", "aug_pointer_grid"),
               "B": ("dag", "grid", "pointer_grid", "aug_pointer_grid", "role_ablated_dag")}
T975_DF4 = 2.7764451051977987


# ----------------------------------------------------------------------------------------------- data
def permute_example(example: CircuitExample, permutation) -> CircuitExample:
    """Rename qubits; operation order and labels are carried over unchanged."""
    ops = [Op(op.gate, tuple(int(permutation[q]) for q in op.wires), op.params) for op in example.ops]
    return CircuitExample(example.n_qubits, ops, example.labels)


def ops_to_circuit(n_qubits: int, ops: list[Op]):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits)
    for op in ops:
        if op.gate == "cx": qc.cx(*op.wires)
        elif op.gate == "h": qc.h(op.wires[0])
        else: getattr(qc, op.gate)(op.params[0], op.wires[0])
    return qc


def build_data(specs=SPLIT_SPECS, relabel_seeds=RELABEL_SEEDS, verify_labels: bool = True) -> dict:
    """Generates every split for both phases. Phase B examples share the Phase A operations and differ in labels."""
    started = time.monotonic(); seen: set[str] = set(); families = {}; wire = {}; comm = {}; checks = {}
    for name, seed, count, qubits in specs:
        examples, names, compiled = generate_suite(seed, count, qubits, exclude=seen, return_compiled=True)
        keys = {examples_hash([x]) for x in examples}
        if len(keys) != len(examples) or keys & seen: raise SystemExit(f"duplicate circuit in {name}")
        seen |= keys; families[name] = names; wire[name] = examples
        labels = [commutation_labels(c) for c in compiled]
        if verify_labels:  # the Op-level circuit must reproduce the labels of the transpiler's own object
            for x, y in zip(examples, labels):
                if not np.array_equal(commutation_labels(ops_to_circuit(x.n_qubits, x.ops)), y): raise SystemExit(f"label mismatch in {name}")
        comm[name] = [CircuitExample(x.n_qubits, x.ops, y) for x, y in zip(examples, labels)]
    permutations = {}
    for variant, base in VARIANTS.items():
        if variant.endswith("relabeled"):
            rng = np.random.default_rng(relabel_seeds[base]); permutations[variant] = [rng.permutation(x.n_qubits) for x in wire[base]]
        else:
            permutations[variant] = [np.arange(x.n_qubits)[::-1] for x in wire[base]]
        families[variant] = families[base]
        wire[variant] = [permute_example(x, p) for x, p in zip(wire[base], permutations[variant])]
        comm[variant] = [permute_example(x, p) for x, p in zip(comm[base], permutations[variant])]
        if verify_labels:  # carried labels must equal labels recomputed on the renamed circuit, for every circuit
            for x in wire[variant]:
                if not np.array_equal(critical_labels(x.n_qubits, x.ops), x.labels): raise SystemExit(f"wire label not invariant in {variant}")
            for x in comm[variant]:
                if not np.array_equal(commutation_labels(ops_to_circuit(x.n_qubits, x.ops)), x.labels): raise SystemExit(f"commutation label not invariant in {variant}")
        checks[variant] = {"circuits": len(wire[variant]), "changed_circuits": int(sum(a.ops != b.ops for a, b in zip(wire[variant], wire[base]))),
                           "labels_recomputed_and_equal": bool(verify_labels)}
    return {"families": families, "A": wire, "B": comm, "checks": checks, "seconds": time.monotonic() - started}


def manifest(data: dict) -> dict:
    stats = {}
    for name in [s[0] for s in SPLIT_SPECS if s[0] in data["A"]] or list(data["A"]):
        if name in VARIANTS: continue
        rows = {}
        for family in sorted(set(data["families"][name])):
            index = [i for i, f in enumerate(data["families"][name]) if f == family]
            a = [data["A"][name][i] for i in index]; b = [data["B"][name][i] for i in index]; nodes = sum(len(x.ops) for x in a)
            rows[family] = {"circuits": len(a), "nodes": nodes, "gates_mean": nodes / len(a), "gates_max": max(len(x.ops) for x in a),
                            "qubits_min": min(x.n_qubits for x in a), "qubits_max": max(x.n_qubits for x in a),
                            "positive_fraction_A": float(sum(x.labels.sum() for x in a) / nodes), "positive_fraction_B": float(sum(x.labels.sum() for x in b) / nodes)}
        every = data["A"][name]; nodes = sum(len(x.ops) for x in every)
        stats[name] = {"circuits": len(every), "nodes": nodes, "gates_mean": nodes / len(every), "gates_max": max(len(x.ops) for x in every),
                       "positive_fraction_A": float(sum(x.labels.sum() for x in every) / nodes),
                       "positive_fraction_B": float(sum(x.labels.sum() for x in data["B"][name]) / nodes),
                       "node_label_agreement_A_B": float(sum((x.labels == y.labels).sum() for x, y in zip(every, data["B"][name])) / nodes), "families": rows}
    import qiskit
    return {"protocol": "QAI_ALGORITHM_SUITE_PROTOCOL_20261114", "families": list(FAMILIES), "split_specs": [list(s[:3]) + [list(s[3])] for s in SPLIT_SPECS],
            "relabel_seeds": RELABEL_SEEDS, "hashes": {phase: {k: examples_hash(v) for k, v in data[phase].items()} for phase in ("A", "B")},
            "statistics": stats, "renaming_checks": data["checks"], "generation_seconds": data["seconds"],
            "versions": {"qiskit": qiskit.__version__, "torch": torch.__version__, "numpy": np.__version__}}


def prepare(output: Path, cache: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True); data = build_data(); record = manifest(data); path = output / "data_manifest.json"
    if path.exists():  # regeneration (for example after losing the cache) must reproduce the committed manifest
        if json.loads(path.read_text())["hashes"] != record["hashes"]: raise SystemExit("regenerated data do not match the committed manifest")
    else:
        path.write_text(json.dumps(record, indent=2) + "\n")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as handle: pickle.dump({"hashes": record["hashes"], "families": data["families"], "A": data["A"], "B": data["B"]}, handle)
    print(json.dumps({k: {m: v[m] for m in ("circuits", "nodes", "positive_fraction_A", "positive_fraction_B")} for k, v in record["statistics"].items()}, indent=1))
    return record


# ----------------------------------------------------------------------------------------------- training
def train(kind: str, training, validation, seed: int, updates: int):
    """The frozen loop (seeding, minibatch stream, loss, checkpointing); `aug` kinds relabel every sampled circuit."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    cls, augment = MODELS[kind]; model = cls()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    pos = sum(float(x.labels.sum()) for x in training); neg = sum(len(x.labels) - float(x.labels.sum()) for x in training)
    weight = torch.tensor(neg / pos); rng = np.random.default_rng(seed); perm_rng = np.random.default_rng(seed + 10000)
    best = -1.; state = None; history = []; started = time.monotonic()
    for update in range(1, updates + 1):
        batch = [training[int(i)] for i in rng.integers(len(training), size=16)]
        if augment: batch = [permute_example(x, perm_rng.permutation(x.n_qubits)) for x in batch]
        loss = torch.stack([F.binary_cross_entropy_with_logits(y, torch.tensor(x.labels), pos_weight=weight)
                            for y, x in zip(model(batch), batch)]).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if update % 250 == 0 or update == updates:
            metrics = evaluate(model, validation); history.append({"update": update, "loss": float(loss.detach()), **metrics})
            if metrics["auc"] > best: best = metrics["auc"]; state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {"kind": kind, "seed": seed, "updates": updates, "parameters": sum(p.numel() for p in model.parameters()),
                   "train_seconds": time.monotonic() - started, "best_validation_auc": best, "history": history}


# ----------------------------------------------------------------------------------------------- evaluation
def auroc(labels: np.ndarray, scores: np.ndarray):
    """Exact Mann-Whitney AUROC with average ranks for ties; None when one class is absent."""
    positive = labels == 1; n_pos = int(positive.sum()); n_neg = len(labels) - n_pos
    if not n_pos or not n_neg: return None
    return float((rankdata(scores)[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def predict(model: nn.Module, examples) -> list[np.ndarray]:
    model.eval(); out = []
    for start in range(0, len(examples), 16):
        out.extend(x.cpu().numpy().astype(np.float64) for x in model(examples[start:start + 16]))
    return out


def split_metrics(logits: list[np.ndarray], examples, families: list[str]) -> dict:
    """AUROC is ranked on float64 logits, so saturated probabilities cannot create ties."""
    score = np.concatenate(logits); truth = np.concatenate([x.labels for x in examples]); predicted = score >= 0.
    owner = np.concatenate([np.full(len(x.labels), i) for i, x in enumerate(examples)]); family_of = np.asarray(families)[owner]
    tp = int(np.sum(predicted & (truth == 1))); fp = int(np.sum(predicted & (truth == 0))); fn = int(np.sum(~predicted & (truth == 1)))
    per = {}
    for family in sorted(set(families)):
        mask = family_of == family
        per[family] = {"auc": auroc(truth[mask], score[mask]), "nodes": int(mask.sum()), "positive_fraction": float(truth[mask].mean())}
    valid = [v["auc"] for v in per.values() if v["auc"] is not None]
    return {"nodes": len(truth), "positive_fraction": float(truth.mean()), "auc": auroc(truth, score), "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
            "accuracy": float(np.mean(predicted == (truth == 1))), "macro_auc": float(np.mean(valid)), "macro_families": len(valid),
            "macro_excluded_single_class": sorted(f for f, v in per.items() if v["auc"] is None), "families": per}


def pair_changes(original: list[np.ndarray], renamed: list[np.ndarray]) -> dict:
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    return {"pairs": len(original), "max_logit_change": float(max(np.max(np.abs(a - b)) for a, b in zip(original, renamed))),
            "mean_abs_probability_change": float(np.mean([np.mean(np.abs(sigmoid(a) - sigmoid(b))) for a, b in zip(original, renamed)]))}


def run_unit(output: Path, cache: Path, phase: str, kind: str, seed: int, updates: int = 4000) -> dict:
    target = output / f"phase{phase}_{kind}_seed{seed}.json"
    if target.exists(): raise SystemExit(f"{target} exists; completed units are never rerun")
    if kind not in PHASE_KINDS[phase] or seed not in SEEDS: raise SystemExit("unit not in protocol")
    with open(cache, "rb") as handle: data = pickle.load(handle)
    committed = json.loads((output / "data_manifest.json").read_text())["hashes"]
    if data["hashes"] != committed: raise SystemExit("cache does not match the committed data manifest")
    parameters = sum(p.numel() for p in MODELS[kind][0]().parameters())
    if parameters != EXPECTED_PARAMETERS[kind]: raise SystemExit(f"{kind}: {parameters} parameters")
    splits = data[phase]; families = data["families"]
    model, info = train(kind, splits["training"], splits["validation"], seed, updates)
    logits = {s: predict(model, splits[s]) for s in TEST}
    result = {"protocol": "QAI_ALGORITHM_SUITE_PROTOCOL_20261114", "phase": phase, "kind": kind, "seed": seed, "torch": torch.__version__,
              "data_hashes_match_manifest": True, "training": info, "splits": {s: split_metrics(logits[s], splits[s], families[s]) for s in TEST},
              "pair_changes": {v: pair_changes(logits[b], logits[v]) for v, b in VARIANTS.items()}}
    temporary = target.with_suffix(".tmp"); temporary.write_text(json.dumps(result, indent=1) + "\n"); os.replace(temporary, target)
    print("DONE", phase, kind, seed, {s: round(result["splits"][s]["macro_auc"], 4) for s in ("same", "near_width", "far_width", "far_relabeled")},
          f"{info['train_seconds']:.0f}s", flush=True)
    return result


# ----------------------------------------------------------------------------------------------- aggregation
def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b; half = T975_DF4 * d.std(ddof=1) / math.sqrt(len(d))
    return {"differences": d.tolist(), "mean": float(d.mean()), "sample_std": float(d.std(ddof=1)), "t95_low": float(d.mean() - half),
            "t95_high": float(d.mean() + half), "first_wins": int(np.sum(d > 0))}


def readings(phase: str, comparisons: dict, aggregate_: dict) -> dict:
    out = {}
    d1 = np.asarray(comparisons["dag_minus_grid"]["differences"]); d2 = np.asarray(comparisons["pointer_grid_minus_grid"]["differences"])
    if np.all(d1 > 0) and np.all(d2 > 0) and d1.mean() >= .03 and d2.mean() >= .03: out["R1_connectivity"] = "replicated"
    elif d1.mean() < .01 or d2.mean() < .01: out["R1_connectivity"] = "not_replicated"
    else: out["R1_connectivity"] = "partial"
    d = np.asarray(comparisons["dag_minus_pointer_grid"]["differences"])
    if abs(d.mean()) < .01: out["R2_dag_vs_pointer"] = "tie"
    elif np.all(d > 0) and d.mean() >= .02: out["R2_dag_vs_pointer"] = "graph_advantage"
    elif np.all(d < 0) and d.mean() <= -.02: out["R2_dag_vs_pointer"] = "sequence_advantage"
    else: out["R2_dag_vs_pointer"] = "intermediate"
    aug = aggregate_["aug_pointer_grid"]; lead = comparisons["dag_minus_aug_pointer_grid"]["mean"]
    gaps = (aug["gaps"]["same_relabeled"]["mean"], aug["gaps"]["far_relabeled"]["mean"])
    if max(gaps) <= .005 and lead <= .01: out["R3_augmentation"] = "suffices"
    elif max(gaps) >= .02 or lead >= .02: out["R3_augmentation"] = "insufficient"
    else: out["R3_augmentation"] = "intermediate"
    worst = max(aggregate_["dag"]["pair_changes"][v]["max_logit_change"] for v in VARIANTS)
    out["R4_dag_invariance"] = "exact_within_1e-6" if worst <= 1e-6 else "FAILED"; out["R4_dag_max_logit_change"] = worst
    if phase == "B":
        d = np.asarray(comparisons["dag_minus_role_ablated_dag"]["differences"])
        out["roles"] = "roles_contribute" if np.all(d >= .01) else "roles_not_needed" if d.mean() < .01 else "intermediate"
    return out


def aggregate(output: Path, phase: str) -> dict:
    units = {kind: [json.loads((output / f"phase{phase}_{kind}_seed{seed}.json").read_text()) for seed in SEEDS] for kind in PHASE_KINDS[phase]}
    agg = {}
    for kind, rows in units.items():
        agg[kind] = {"parameters": rows[0]["training"]["parameters"], "train_seconds": [r["training"]["train_seconds"] for r in rows], "splits": {}, "gaps": {}, "pair_changes": {}}
        for s in TEST:
            entry = {}
            for metric in ("macro_auc", "auc", "f1", "accuracy"):
                v = np.asarray([r["splits"][s][metric] for r in rows])
                entry.update({metric + "_mean": float(v.mean()), metric + "_sample_std": float(v.std(ddof=1)), metric + "_values": v.tolist()})
            entry["macro_excluded_single_class"] = rows[0]["splits"][s]["macro_excluded_single_class"]
            entry["family_auc_mean"] = {f: (None if rows[0]["splits"][s]["families"][f]["auc"] is None else float(np.mean([r["splits"][s]["families"][f]["auc"] for r in rows])))
                                        for f in rows[0]["splits"][s]["families"]}
            agg[kind]["splits"][s] = entry
        for variant, base in VARIANTS.items():
            g = np.asarray([r["splits"][base]["macro_auc"] - r["splits"][variant]["macro_auc"] for r in rows])
            agg[kind]["gaps"][variant] = {"values": g.tolist(), "mean": float(g.mean()), "sample_std": float(g.std(ddof=1))}
            agg[kind]["pair_changes"][variant] = {"max_logit_change": float(max(r["pair_changes"][variant]["max_logit_change"] for r in rows)),
                                                  "mean_abs_probability_change": float(np.mean([r["pair_changes"][variant]["mean_abs_probability_change"] for r in rows]))}
    far = {kind: np.asarray(agg[kind]["splits"]["far_width"]["macro_auc_values"]) for kind in units}
    pairs = [("dag", "grid"), ("pointer_grid", "grid"), ("dag", "pointer_grid"), ("dag", "aug_pointer_grid")] + ([("dag", "role_ablated_dag")] if phase == "B" else [])
    comparisons = {f"{a}_minus_{b}": paired(far[a], far[b]) for a, b in pairs}
    result = {"protocol": "QAI_ALGORITHM_SUITE_PROTOCOL_20261114 (post-confirmatory)", "phase": phase, "seeds": list(SEEDS), "updates_per_model": 4000,
              "label": "wire-dependency critical path" if phase == "A" else "commutation-aware critical chain (Qiskit DAGDependency)",
              "primary_metric": "far_width family-macro node AUROC", "data_hashes": json.loads((output / "data_manifest.json").read_text())["hashes"][phase],
              "aggregate": agg, "comparisons_far_macro_auc": comparisons, "readings": readings(phase, comparisons, agg),
              "status": "post_confirmatory_not_part_of_primary_gate"}
    (output / f"phase{phase}_result.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"readings": result["readings"], "far_macro_auc": {k: round(v["splits"]["far_width"]["macro_auc_mean"], 4) for k, v in agg.items()},
                      "comparisons": {k: [round(v["mean"], 4), v["first_wins"]] for k, v in comparisons.items()}}, indent=1))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "unit", "aggregate"))
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--cache", type=Path)
    parser.add_argument("--phase", choices=("A", "B")); parser.add_argument("--kind", choices=list(MODELS)); parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args.output, args.cache)
    elif args.command == "unit": run_unit(args.output, args.cache, args.phase, args.kind, args.seed)
    else: aggregate(args.output, args.phase)


if __name__ == "__main__": main()
