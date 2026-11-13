"""Post-confirmatory relabeling-augmentation control for the QAI study.

Protocol: experiments/QAI_AUGMENTATION_CONTROL_PROTOCOL_20261113.md (frozen before this code).
Frozen modules are imported, never modified.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .connectivity_controls import FROZEN_FINAL, SEEDS, PointerGrid, frozen_splits
from .invariant_benchmark import InvariantGrid, evaluate, relabel
from .schedule_benchmark import examples_hash

MODELS = {"aug_pointer_grid": PointerGrid, "aug_grid": InvariantGrid}
EXPECTED_PARAMETERS = {"aug_pointer_grid": 87457, "aug_grid": 84961}
TEST = ("same", "near_width", "far_width", "renamed", "far_renamed")


def train_augmented(kind: str, training, validation, seed: int, updates: int):
    """invariant_benchmark.train with each sampled circuit replaced by a fresh random relabeling."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    model = MODELS[kind]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    pos = sum(float(x.labels.sum()) for x in training); neg = sum(len(x.labels) - float(x.labels.sum()) for x in training)
    weight = torch.tensor(neg / pos); rng = np.random.default_rng(seed); perm_rng = np.random.default_rng(seed + 10000)
    best = -1.; state = None; history = []; started = time.monotonic()
    for update in range(1, updates + 1):
        indices = rng.integers(len(training), size=16)  # identical stream to the frozen runs
        batch = [relabel(training[int(i)], perm_rng) for i in indices]
        loss = torch.stack([F.binary_cross_entropy_with_logits(y, torch.tensor(x.labels), pos_weight=weight)
                            for y, x in zip(model(batch), batch)]).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if update % 250 == 0 or update == updates:
            metrics = evaluate(model, validation); history.append({"update": update, "loss": float(loss.detach()), **metrics})
            if metrics["auc"] > best: best = metrics["auc"]; state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {"kind": kind, "seed": seed, "updates": updates, "parameters": sum(p.numel() for p in model.parameters()),
                   "train_seconds": time.monotonic() - started, "best_validation_auc": best, "history": history}


@torch.no_grad()
def pair_changes(model, originals, changed) -> dict:
    model.eval(); prob = []; logit = []
    for a, b in zip(originals, changed):
        la, lb = model([a])[0], model([b])[0]
        logit.append(float(torch.max(torch.abs(la - lb)))); prob.append(float(torch.mean(torch.abs(torch.sigmoid(la) - torch.sigmoid(lb)))))
    return {"mean_abs_probability_change": float(np.mean(prob)), "max_logit_change": float(max(logit))}


def run_kind(kind: str, output: Path, updates: int = 4000) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    splits = frozen_splits(); recorded = json.loads(FROZEN_FINAL.read_text())
    hashes = {k: examples_hash(v) for k, v in splits.items()}
    if hashes != recorded["dataset_hashes"]: raise SystemExit(f"dataset hash mismatch: {hashes}")
    n = sum(p.numel() for p in MODELS[kind]().parameters())
    if n != EXPECTED_PARAMETERS[kind]: raise SystemExit(f"{kind}: {n} parameters, expected {EXPECTED_PARAMETERS[kind]}")
    frng = np.random.default_rng(14106); splits["far_renamed"] = [relabel(x, frng) for x in splits["far_width"]]
    rows = []
    for seed in SEEDS:
        model, info = train_augmented(kind, splits["training"], splits["validation"], seed, updates)
        row = {"seed": seed, "training": info, **{s: evaluate(model, splits[s]) for s in TEST},
               "same_pair_changes": pair_changes(model, splits["same"][:64], splits["renamed"][:64]),
               "far_pair_changes": pair_changes(model, splits["far_width"][:64], splits["far_renamed"][:64])}
        rows.append(row)
        (output / f"{kind}_partial.json").write_text(json.dumps(rows) + "\n")
        print(kind, seed, {s: round(row[s]["auc"], 4) for s in TEST}, f"{info['train_seconds']:.0f}s", flush=True)
    result = {"kind": kind, "far_renamed_hash": examples_hash(splits["far_renamed"]), "dataset_hashes": hashes, "seed_results": rows}
    (output / f"{kind}_result.json").write_text(json.dumps(result, indent=2) + "\n"); (output / f"{kind}_partial.json").unlink()
    return result


def aggregate(output: Path) -> dict:
    recorded = json.loads(FROZEN_FINAL.read_text()); rec = {r["seed"]: r["models"] for r in recorded["seed_results"]}
    out = {"protocol": "QAI_AUGMENTATION_CONTROL_PROTOCOL_20261113 (post-confirmatory)", "seeds": list(SEEDS), "torch": torch.__version__,
           "status": "post_confirmatory_not_part_of_primary_gate", "aggregate": {}, "comparisons": {}, "seed_results": {}}
    for kind in MODELS:
        r = json.loads((output / f"{kind}_result.json").read_text()); rows = r["seed_results"]
        out["far_renamed_hash"] = r["far_renamed_hash"]; out["dataset_hashes"] = r["dataset_hashes"]; out["seed_results"][kind] = rows
        agg = {}
        for s in TEST:
            agg[s] = {}
            for m in ("auc", "f1", "accuracy"):
                v = np.asarray([x[s][m] for x in rows])
                agg[s].update({m + "_mean": float(v.mean()), m + "_sample_std": float(v.std(ddof=1)), m + "_values": v.tolist()})
        same_gap = np.asarray([x["same"]["auc"] - x["renamed"]["auc"] for x in rows])
        far_gap = np.asarray([x["far_width"]["auc"] - x["far_renamed"]["auc"] for x in rows])
        d = np.asarray([rec[x["seed"]]["dag"]["far_width"]["auc"] - x["far_width"]["auc"] for x in rows])
        agg["same_relabel_gap_mean"] = float(same_gap.mean()); agg["far_relabel_gap_mean"] = float(far_gap.mean())
        for key in ("same_pair_changes", "far_pair_changes"):
            agg[key] = {"mean_abs_probability_change": float(np.mean([x[key]["mean_abs_probability_change"] for x in rows])),
                        "max_logit_change": float(max(x[key]["max_logit_change"] for x in rows))}
        out["aggregate"][kind] = agg
        out["comparisons"][kind] = {"recorded_dag_minus_model_far_auc": d.tolist(), "mean": float(d.mean()), "sample_std": float(d.std(ddof=1)),
                                    "t95_half_width": float(2.7764451051977987 * d.std(ddof=1) / np.sqrt(len(d))), "dag_wins": int(np.sum(d > 0))}
    a = out["aggregate"]["aug_pointer_grid"]; dm = out["comparisons"]["aug_pointer_grid"]["mean"]
    if a["same_relabel_gap_mean"] <= .005 and a["far_relabel_gap_mean"] <= .005 and dm <= .01: reading = "augmentation_suffices"
    elif a["same_relabel_gap_mean"] >= .02 or a["far_relabel_gap_mean"] >= .02 or dm >= .02: reading = "augmentation_insufficient"
    else: reading = "intermediate"
    out["reading_aug_pointer_grid"] = reading
    (output / "augmentation_result.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"reading": reading, **{k: {"far": v["far_width"]["auc_mean"], "same_gap": v["same_relabel_gap_mean"], "far_gap": v["far_relabel_gap_mean"]}
                                             for k, v in out["aggregate"].items()}, "dag_minus": {k: v["mean"] for k, v in out["comparisons"].items()}}, indent=2))
    return out


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=list(MODELS)); parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--updates", type=int, default=4000); args = parser.parse_args()
    if args.aggregate: aggregate(args.output)
    else: run_kind(args.kind, args.output, args.updates)


if __name__ == "__main__": main()
