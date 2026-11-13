"""Post-confirmatory index-free connectivity controls for the QAI study.

Protocol: experiments/QAI_CONNECTIVITY_CONTROLS_PROTOCOL_20261113.md (frozen before this code).
The frozen modules (invariant_benchmark.py) are imported, never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .invariant_benchmark import (InvariantDag, InvariantGrid, InvariantTransformer, evaluate, invariant_node_features,
                                  relabel, role_edges)
from .schedule_benchmark import CircuitExample, canonical_order, examples_hash
from .structured_transfer import generate_structured

POINTER_FEATURES = 33
MAX_DISTANCE = 192
FROZEN_FINAL = Path("experiments/qai_invariant_final_20261102/final_result.json")
SEEDS = (111, 112, 113, 114, 115)


def operand_role(op, wire: int) -> int:
    if op.gate != "cx": return 0
    return 1 if op.wires[0] == wire else 2


def pointer_features(example: CircuitExample, order: list[int] | None = None) -> np.ndarray:
    """33 features per op, rows in canonical order. Wire ids locate neighbours only; never emitted."""
    order = canonical_order(example) if order is None else order
    position = {node: p for p, node in enumerate(order)}
    chains: dict[int, list[int]] = {}
    for node in order:  # canonical order is topological, so each chain is in program order
        for wire in example.ops[node].wires: chains.setdefault(int(wire), []).append(node)
    neighbours = {}
    for wire, chain in chains.items():
        for k, node in enumerate(chain):
            neighbours[(node, wire)] = (chain[k - 1] if k else None, chain[k + 1] if k + 1 < len(chain) else None)
    base = invariant_node_features(example)
    scale = math.log1p(MAX_DISTANCE)
    rows = []
    for node in order:
        op = example.ops[node]; row = list(base[node])
        for slot in range(2):
            if slot >= len(op.wires): row.extend([0.0] * 13); continue
            wire = int(op.wires[slot]); role = operand_role(op, wire)
            row.extend(float(role == r) for r in range(3))
            before, after = neighbours[(node, wire)]
            for other in (before, after):
                if other is None: row.extend([0.0] * 5); continue
                other_role = operand_role(example.ops[other], wire)
                row.append(1.0); row.extend(float(other_role == r) for r in range(3))
                row.append(math.log1p(abs(position[node] - position[other])) / scale)
        rows.append(row)
    return np.asarray(rows, np.float32).reshape(-1, POINTER_FEATURES)


def _restore(logits: torch.Tensor, order: list[int]) -> torch.Tensor:
    restored = torch.empty_like(logits); restored[torch.tensor(order, device=logits.device)] = logits; return restored


class PointerGrid(InvariantGrid):
    def __init__(self, width: int = 96):
        super().__init__(width); self.node_in = nn.Linear(POINTER_FEATURES, width)

    def forward(self, examples: list[CircuitExample]) -> list[torch.Tensor]:
        device = self.node_in.weight.device; orders = [canonical_order(x) for x in examples]
        feats = [pointer_features(x, o) for x, o in zip(examples, orders)]; lengths = [len(f) for f in feats]
        padded = torch.zeros(len(examples), max(lengths), POINTER_FEATURES, device=device)
        for i, f in enumerate(feats): padded[i, :len(f)] = torch.tensor(f, device=device)
        packed = nn.utils.rnn.pack_padded_sequence(F.gelu(self.node_in(padded)), lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed); unpacked, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        return [_restore(self.out(unpacked[i, :n]).squeeze(1), o) for i, (n, o) in enumerate(zip(lengths, orders))]


class PointerTransformer(InvariantTransformer):
    def __init__(self, width: int = 64, max_length: int = MAX_DISTANCE):
        super().__init__(width, max_length); self.node_in = nn.Linear(POINTER_FEATURES, width)

    def forward(self, examples: list[CircuitExample]) -> list[torch.Tensor]:
        device = self.node_in.weight.device; orders = [canonical_order(x) for x in examples]
        feats = [pointer_features(x, o) for x, o in zip(examples, orders)]; lengths = [len(f) for f in feats]; maximum = max(lengths)
        padded = torch.zeros(len(examples), maximum, POINTER_FEATURES, device=device)
        mask = torch.ones(len(examples), maximum, dtype=torch.bool, device=device)
        for i, f in enumerate(feats): padded[i, :len(f)] = torch.tensor(f, device=device); mask[i, :len(f)] = False
        h = F.gelu(self.node_in(padded)) + self.position(torch.arange(maximum, device=device))[None]
        encoded = self.encoder(h, src_key_padding_mask=mask)
        return [_restore(self.out(encoded[i, :n]).squeeze(1), o) for i, (n, o) in enumerate(zip(lengths, orders))]


class RoleAblatedDag(InvariantDag):
    """InvariantDag whose edge attributes are all zero: topology only."""

    def forward_one(self, example: CircuitExample) -> torch.Tensor:
        device = self.node_in.weight.device
        h = F.gelu(self.node_in(torch.tensor(invariant_node_features(example), device=device)))
        edge_np, attr_np = role_edges(example)
        edge = torch.tensor(edge_np, dtype=torch.long, device=device); attr = torch.zeros(attr_np.shape, device=device)
        for fwd, bwd, self_layer, norm in zip(self.fwd, self.bwd, self.self_layers, self.norms):
            af = torch.zeros_like(h); ab = torch.zeros_like(h)
            if edge.shape[1]:
                af.index_add_(0, edge[1], fwd(torch.cat([h[edge[0]], attr], 1)))
                ab.index_add_(0, edge[0], bwd(torch.cat([h[edge[1]], attr], 1)))
            h = norm(h + F.gelu(self_layer(h) + af + ab))
        return self.out(h).squeeze(1)


MODELS = {"pointer_grid": PointerGrid, "pointer_transformer": PointerTransformer, "role_ablated_dag": RoleAblatedDag}
EXPECTED_PARAMETERS = {"pointer_grid": 87457, "pointer_transformer": 89729, "role_ablated_dag": 80833}


def train(kind: str, training, validation, seed: int, updates: int):
    """Same loop as invariant_benchmark.train (seeding, minibatch stream, loss, checkpointing)."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    model = MODELS[kind]()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    pos = sum(float(x.labels.sum()) for x in training); neg = sum(len(x.labels) - float(x.labels.sum()) for x in training)
    weight = torch.tensor(neg / pos); rng = np.random.default_rng(seed); best = -1.; state = None; history = []; started = time.monotonic()
    for update in range(1, updates + 1):
        batch = [training[int(i)] for i in rng.integers(len(training), size=16)]
        loss = torch.stack([F.binary_cross_entropy_with_logits(y, torch.tensor(x.labels), pos_weight=weight)
                            for y, x in zip(model(batch), batch)]).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        if update % 250 == 0 or update == updates:
            metrics = evaluate(model, validation); history.append({"update": update, "loss": float(loss.detach()), **metrics})
            if metrics["auc"] > best: best = metrics["auc"]; state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    return model, {"kind": kind, "seed": seed, "updates": updates, "parameters": sum(p.numel() for p in model.parameters()),
                   "train_seconds": time.monotonic() - started, "best_validation_auc": best, "history": history}


def frozen_splits():
    fam = ("qaoa", "vqe", "arithmetic")
    training = generate_structured(14100, 1000, (4, 7), (32, 64), fam); validation = generate_structured(14101, 256, (4, 7), (32, 64), fam)
    same = generate_structured(14102, 512, (4, 7), (32, 64), fam); near = generate_structured(14103, 512, (8, 12), (72, 104), fam)
    far = generate_structured(14104, 512, (13, 20), (96, 144), fam); rng = np.random.default_rng(14105); renamed = [relabel(x, rng) for x in same]
    return {"training": training, "validation": validation, "same": same, "near_width": near, "far_width": far, "renamed": renamed}


def band(diffs: np.ndarray) -> str:
    if np.all(diffs > 0) and diffs.mean() >= .02: return "topology_supported"
    if diffs.mean() < .01 or np.sum(diffs < 0) >= 3: return "information_explains_gap"
    return "intermediate"


def run(output: Path, updates: int = 4000, kinds=tuple(MODELS)) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    splits = frozen_splits(); recorded = json.loads(FROZEN_FINAL.read_text())
    hashes = {k: examples_hash(v) for k, v in splits.items()}
    if hashes != recorded["dataset_hashes"]: raise SystemExit(f"dataset hash mismatch: {hashes}")
    for kind in kinds:
        n = sum(p.numel() for p in MODELS[kind]().parameters())
        if n != EXPECTED_PARAMETERS[kind]: raise SystemExit(f"{kind}: {n} parameters, expected {EXPECTED_PARAMETERS[kind]}")
    rec_seed = {r["seed"]: r["models"] for r in recorded["seed_results"]}
    test = ("same", "near_width", "far_width", "renamed"); rows = {k: [] for k in kinds}
    for seed in SEEDS:
        for kind in kinds:
            model, info = train(kind, splits["training"], splits["validation"], seed, updates)
            row = {"seed": seed, "training": info, **{s: evaluate(model, splits[s]) for s in test}}
            with torch.no_grad():
                row["max_relabel_logit_difference_first64"] = max(float(torch.max(torch.abs(model([a])[0] - model([b])[0])))
                                                                   for a, b in zip(splits["same"][:64], splits["renamed"][:64]))
            rows[kind].append(row)
            partial = {"status": "running", "rows": rows}; (output / "partial.json").write_text(json.dumps(partial) + "\n")
            print(kind, seed, {s: round(row[s]["auc"], 4) for s in test}, f"{info['train_seconds']:.0f}s", flush=True)
    aggregate = {}; comparisons = {}
    for kind in kinds:
        aggregate[kind] = {}
        for s in test:
            aggregate[kind][s] = {}
            for m in ("auc", "f1", "accuracy"):
                v = np.asarray([r[s][m] for r in rows[kind]])
                aggregate[kind][s].update({m + "_mean": float(v.mean()), m + "_sample_std": float(v.std(ddof=1)), m + "_values": v.tolist()})
        dag_far = np.asarray([rec_seed[r["seed"]]["dag"]["far_width"]["auc"] for r in rows[kind]])
        grid_far = np.asarray([rec_seed[r["seed"]]["grid"]["far_width"]["auc"] for r in rows[kind]])
        own_far = np.asarray([r["far_width"]["auc"] for r in rows[kind]])
        d = dag_far - own_far
        comparisons[kind] = {"recorded_dag_minus_model_far_auc": d.tolist(), "mean": float(d.mean()), "sample_std": float(d.std(ddof=1)),
                             "t95_half_width": float(2.7764451051977987 * d.std(ddof=1) / math.sqrt(len(d))),
                             "dag_wins": int(np.sum(d > 0)), "model_minus_recorded_grid_far_auc": (own_far - grid_far).tolist()}
    readings = {}
    if "pointer_grid" in comparisons: readings["pointer_grid_H1"] = band(np.asarray(comparisons["pointer_grid"]["recorded_dag_minus_model_far_auc"]))
    if "pointer_transformer" in comparisons: readings["pointer_transformer_descriptive"] = band(np.asarray(comparisons["pointer_transformer"]["recorded_dag_minus_model_far_auc"]))
    if "role_ablated_dag" in comparisons:
        d = np.asarray(comparisons["role_ablated_dag"]["recorded_dag_minus_model_far_auc"])
        readings["role_ablated_dag_descriptive"] = ("roles_not_needed" if abs(d.mean()) < .01 else
                                                    "roles_contribute" if np.all(d >= .01) else "intermediate")
    result = {"protocol": "QAI_CONNECTIVITY_CONTROLS_PROTOCOL_20261113 (post-confirmatory)", "updates_per_model": updates,
              "seeds": list(SEEDS), "dataset_hashes": hashes, "dataset_hashes_match_frozen_final": True,
              "torch": torch.__version__, "seed_results": rows, "aggregate": aggregate, "comparisons": comparisons,
              "readings": readings, "status": "post_confirmatory_not_part_of_primary_gate"}
    (output / "controls_result.json").write_text(json.dumps(result, indent=2) + "\n"); (output / "partial.json").unlink()
    print(json.dumps({"readings": readings, "comparisons": {k: {kk: v[kk] for kk in ("mean", "dag_wins")} for k, v in comparisons.items()}}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=4000); parser.add_argument("--kinds", nargs="+", default=list(MODELS))
    args = parser.parse_args(); run(args.output, args.updates, tuple(args.kinds))


if __name__ == "__main__": main()
