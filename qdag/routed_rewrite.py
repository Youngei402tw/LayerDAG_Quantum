"""Rank safe rewrite templates by downstream hardware-routing utility."""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import Op, resources, simulate
from .schedule_benchmark import (
    MAX_QUBITS, CanonicalGridCriticalModel, CircuitExample, DagCriticalModel,
    canonical_order, critical_labels, dependency_edges, node_features,
)


@dataclass
class RewriteCandidate:
    indices: tuple[int, ...]
    template: str


@dataclass
class RoutedRewriteExample:
    circuit: CircuitExample
    candidates: list[RewriteCandidate]
    utilities: np.ndarray | None = None  # columns: routed CX reduction, routed depth reduction
    family: str = ""


def _qaoa(rng: np.random.Generator, n: int, length: int) -> list[Op]:
    ops = [Op("h", (q,)) for q in range(n)]
    while len(ops) < length:
        for q in range(n):
            r = (q + 1 + int(rng.integers(max(1, n - 1)))) % n
            if r == q:
                r = (q + 1) % n
            theta = float(rng.uniform(-math.pi, math.pi))
            ops.extend([Op("cx", (q, r)), Op("rz", (r,), (theta,)), Op("cx", (q, r))])
        for q in range(n):
            beta = float(rng.uniform(-math.pi, math.pi))
            ops.extend([Op("h", (q,)), Op("rz", (q,), (beta,)), Op("h", (q,))])
    return ops[:length]


def _vqe(rng: np.random.Generator, n: int, length: int) -> list[Op]:
    ops: list[Op] = []
    while len(ops) < length:
        for q in range(n):
            ops.extend([Op("ry", (q,), (float(rng.uniform(-math.pi, math.pi)),)),
                        Op("rz", (q,), (float(rng.uniform(-math.pi, math.pi)),))])
        offset = int(rng.integers(1, n))
        for q in range(n):
            r = (q + offset) % n
            if q != r:
                ops.append(Op("cx", (q, r)))
    return ops[:length]


def _arithmetic(rng: np.random.Generator, n: int, length: int) -> list[Op]:
    ops: list[Op] = []
    while len(ops) < length:
        pivot = int(rng.integers(n))
        order = rng.permutation(n).tolist()
        for q in order:
            if q != pivot:
                ops.append(Op("cx", (pivot, int(q))))
        for q in reversed(order):
            if q != pivot and rng.random() < 0.55:
                ops.append(Op("cx", (int(q), pivot)))
        ops.append(Op("rz", (pivot,), (float(rng.uniform(-math.pi, math.pi)),)))
    return ops[:length]


def _chemistry(rng: np.random.Generator, n: int, length: int) -> list[Op]:
    ops: list[Op] = []
    while len(ops) < length:
        p, q = rng.choice(n, 2, replace=False).tolist()
        theta = float(rng.uniform(-math.pi, math.pi))
        ops.extend([Op("cx", (int(p), int(q))), Op("ry", (int(q),), (theta,)),
                    Op("cx", (int(p), int(q))), Op("rz", (int(p),), (-theta / 2,))])
    return ops[:length]


FAMILIES = {"qaoa": _qaoa, "vqe": _vqe, "arithmetic": _arithmetic, "chemistry": _chemistry}


def _insert_candidates(rng: np.random.Generator, n: int, base: list[Op], count: int = 7) -> tuple[list[Op], list[RewriteCandidate]]:
    ops = list(base)
    candidates: list[RewriteCandidate] = []
    for k in range(count):
        pos = int(rng.integers(len(ops) + 1))
        q = int(rng.integers(n))
        kind = k % 3
        if kind == 0:
            block = [Op("rz", (q,), (0.0,))]
            template = "bound_zero_rotation"
        elif kind == 1:
            theta = float(rng.uniform(0.15, math.pi))
            block = [Op("rz", (q,), (theta,)), Op("rz", (q,), (-theta,))]
            template = "inverse_rotation_pair"
        else:
            r = int(rng.integers(n - 1))
            if r >= q:
                r += 1
            block = [Op("cx", (q, r)), Op("cx", (q, r))]
            template = "self_inverse_cx_pair"
        shift = len(block)
        ops[pos:pos] = block
        candidates = [RewriteCandidate(tuple(i + shift if i >= pos else i for i in c.indices), c.template) for c in candidates]
        candidates.append(RewriteCandidate(tuple(range(pos, pos + shift)), template))
    return ops, candidates


def generate_family_examples(seed: int, count: int, qubits: tuple[int, int], lengths: tuple[int, int],
                             families: tuple[str, ...]) -> list[RoutedRewriteExample]:
    rng = np.random.default_rng(seed)
    output = []
    for i in range(count):
        family = families[i % len(families)]
        n = int(rng.integers(qubits[0], qubits[1] + 1))
        length = int(rng.integers(lengths[0], lengths[1] + 1))
        base = FAMILIES[family](rng, n, length)
        ops, candidates = _insert_candidates(rng, n, base)
        output.append(RoutedRewriteExample(CircuitExample(n, ops, critical_labels(n, ops)), candidates, family=family))
    return output


def remove_candidate(example: RoutedRewriteExample, candidate: RewriteCandidate) -> list[Op]:
    removed = set(candidate.indices)
    return [op for i, op in enumerate(example.circuit.ops) if i not in removed]


def _qiskit_metrics(n: int, ops: list[Op]) -> tuple[int, int]:
    from qiskit import QuantumCircuit, transpile
    from qiskit.transpiler import CouplingMap
    qc = QuantumCircuit(n)
    for op in ops:
        if op.gate == "h": qc.h(op.wires[0])
        elif op.gate == "ry": qc.ry(op.params[0], op.wires[0])
        elif op.gate == "rz": qc.rz(op.params[0], op.wires[0])
        elif op.gate == "cx": qc.cx(*op.wires)
        else: raise ValueError(op.gate)
    coupling = CouplingMap([(q, q + 1) for q in range(n - 1)] + [(q + 1, q) for q in range(n - 1)])
    routed = transpile(qc, basis_gates=["rz", "sx", "x", "cx"], coupling_map=coupling,
                       initial_layout=list(range(n)), optimization_level=0, seed_transpiler=17)
    return int(routed.count_ops().get("cx", 0)), int(routed.depth())


def label_utilities(examples: list[RoutedRewriteExample], verify: bool = True) -> None:
    for example in examples:
        base = _qiskit_metrics(example.circuit.n_qubits, example.circuit.ops)
        values = []
        reference = simulate(example.circuit.n_qubits, example.circuit.ops) if verify else None
        for candidate in example.candidates:
            edited = remove_candidate(example, candidate)
            if verify:
                fidelity = abs(np.vdot(reference, simulate(example.circuit.n_qubits, edited))) ** 2
                if fidelity < 1 - 1e-10:
                    raise AssertionError((candidate.template, fidelity))
            metric = _qiskit_metrics(example.circuit.n_qubits, edited)
            values.append((base[0] - metric[0], base[1] - metric[1]))
        example.utilities = np.asarray(values, np.float32)


def _candidate_logits(logits: torch.Tensor, example: RoutedRewriteExample) -> torch.Tensor:
    return torch.stack([logits[list(c.indices)].mean() for c in example.candidates])


class CandidateAwareRanker(nn.Module):
    """Matched candidate readout on DAG or canonical-grid contextual node states."""
    def __init__(self, kind: str, width: int = 64, layers: int = 6):
        super().__init__(); self.kind = kind; self.width = width
        input_size = 4 + 3 * MAX_QUBITS + 3
        self.node_in = nn.Linear(input_size, width)
        if kind == "dag":
            self.forward_layers = nn.ModuleList([nn.Linear(width + MAX_QUBITS, width) for _ in range(layers)])
            self.backward_layers = nn.ModuleList([nn.Linear(width + MAX_QUBITS, width) for _ in range(layers)])
            self.self_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        elif kind == "canonical_grid":
            self.gru = nn.GRU(width, width // 2, num_layers=2, bidirectional=True, batch_first=True)
        else: raise ValueError(kind)
        # candidate mean/max, circuit mean, template one-hot, normalized position/size
        self.readout = nn.Sequential(nn.Linear(width * 3 + 5, 96), nn.GELU(), nn.Linear(96, 1))

    def encode(self, example: RoutedRewriteExample) -> torch.Tensor:
        device = self.node_in.weight.device
        x = F.gelu(self.node_in(torch.tensor(node_features(example.circuit), device=device)))
        if self.kind == "dag":
            edge_np, wire_np = dependency_edges(example.circuit.n_qubits, example.circuit.ops)
            edge = torch.tensor(edge_np, dtype=torch.long, device=device)
            attr = F.one_hot(torch.tensor(wire_np, device=device), MAX_QUBITS).float()
            h = x
            for fwd, bwd, self_layer, norm in zip(self.forward_layers, self.backward_layers, self.self_layers, self.norms):
                af = torch.zeros_like(h); ab = torch.zeros_like(h)
                if edge.shape[1]:
                    af.index_add_(0, edge[1], fwd(torch.cat([h[edge[0]], attr], 1)))
                    ab.index_add_(0, edge[0], bwd(torch.cat([h[edge[1]], attr], 1)))
                h = norm(h + F.gelu(self_layer(h) + af + ab))
            return h
        order = canonical_order(example.circuit)
        encoded, _ = self.gru(x[torch.tensor(order, device=device)][None])
        restored = torch.empty_like(encoded[0]); restored[torch.tensor(order, device=device)] = encoded[0]
        return restored

    def forward_one(self, example: RoutedRewriteExample) -> torch.Tensor:
        h = self.encode(example); global_mean = h.mean(0); rows = []
        names = ("bound_zero_rotation", "inverse_rotation_pair", "self_inverse_cx_pair")
        for candidate in example.candidates:
            selected = h[list(candidate.indices)]
            template = [float(candidate.template == name) for name in names]
            extra = torch.tensor(template + [candidate.indices[0] / max(1, len(example.circuit.ops) - 1), len(candidate.indices) / 2], device=h.device)
            rows.append(torch.cat([selected.mean(0), selected.amax(0), global_mean, extra]))
        return self.readout(torch.stack(rows)).squeeze(1)

    def forward(self, examples: list[RoutedRewriteExample]) -> list[torch.Tensor]:
        return [self.forward_one(example) for example in examples]


def train_ranker(kind: str, training: list[RoutedRewriteExample], validation: list[RoutedRewriteExample],
                 seed: int, updates: int = 2000, candidate_aware: bool = True) -> tuple[torch.nn.Module, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CandidateAwareRanker(kind, width=64 if kind == "dag" else 88) if candidate_aware else (DagCriticalModel() if kind == "dag" else CanonicalGridCriticalModel(width=96))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    best, best_state = -math.inf, None
    started = time.perf_counter()
    history = []
    for update in range(1, updates + 1):
        batch = [training[int(i)] for i in rng.integers(len(training), size=16)]
        losses = []
        inputs = batch if candidate_aware else [x.circuit for x in batch]
        for logits, example in zip(model(inputs), batch):
            utility = example.utilities
            values = torch.tensor(utility[:, 0] + 0.05 * utility[:, 1], dtype=torch.float32)
            target = F.softmax(values / 2.0, dim=0)
            scores = logits if candidate_aware else _candidate_logits(logits, example)
            losses.append(-(target * F.log_softmax(scores, dim=0)).sum())
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if update % 250 == 0 or update == updates:
            metric = evaluate(model, validation)["mean_cx_reduction"]
            history.append({"update": update, "loss": float(loss.detach()), "validation_mean_cx_reduction": metric})
            if metric > best:
                best = metric
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"kind": kind, "seed": seed, "updates": updates, "parameters": sum(p.numel() for p in model.parameters()),
                   "train_seconds": time.perf_counter() - started, "best_validation_mean_cx_reduction": best, "history": history}


@torch.no_grad()
def model_scores(model: torch.nn.Module, examples: list[RoutedRewriteExample]) -> list[np.ndarray]:
    model.eval(); output = []
    for example in examples:
        if isinstance(model, CandidateAwareRanker):
            output.append(model([example])[0].cpu().numpy())
        else:
            logits = model([example.circuit])[0]
            output.append(_candidate_logits(logits, example).cpu().numpy())
    return output


def heuristic_scores(examples: list[RoutedRewriteExample], kind: str, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed); result = []
    for example in examples:
        if kind == "random": score = rng.random(len(example.candidates))
        elif kind == "first": score = -np.arange(len(example.candidates), dtype=float)
        elif kind == "source_depth":
            base = resources(example.circuit.n_qubits, example.circuit.ops)["depth"]
            score = np.array([base - resources(example.circuit.n_qubits, remove_candidate(example, c))["depth"] for c in example.candidates])
        elif kind == "critical_path":
            labels = critical_labels(example.circuit.n_qubits, example.circuit.ops)
            score = np.array([labels[list(c.indices)].mean() for c in example.candidates])
        else: raise ValueError(kind)
        result.append(score)
    return result


def evaluate_scores(examples: list[RoutedRewriteExample], scores: list[np.ndarray]) -> dict:
    chosen, oracle, exact = [], [], 0
    for example, score in zip(examples, scores):
        utility = example.utilities
        index = int(np.argmax(score))
        oracle_index = int(np.lexsort((utility[:, 1], utility[:, 0]))[-1])
        chosen.append(utility[index]); oracle.append(utility[oracle_index]); exact += index == oracle_index
    chosen = np.asarray(chosen); oracle = np.asarray(oracle)
    oracle_total = oracle[:, 0].sum()
    return {"circuits": len(examples), "mean_cx_reduction": float(chosen[:, 0].mean()),
            "mean_depth_reduction": float(chosen[:, 1].mean()), "exact_oracle_choice_rate": exact / len(examples),
            "oracle_mean_cx_reduction": float(oracle[:, 0].mean()), "oracle_mean_depth_reduction": float(oracle[:, 1].mean()),
            "cx_recovery": float(chosen[:, 0].sum() / max(1.0, oracle_total)),
            "mean_cx_regret": float((oracle[:, 0] - chosen[:, 0]).mean())}


def evaluate(model: torch.nn.Module, examples: list[RoutedRewriteExample]) -> dict:
    return evaluate_scores(examples, model_scores(model, examples))


def run_development(output: Path, updates: int = 1500) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    training = generate_family_examples(11001, 240, (4, 7), (32, 64), ("qaoa", "vqe", "arithmetic"))
    validation = generate_family_examples(11002, 72, (4, 7), (32, 64), ("qaoa", "vqe", "arithmetic"))
    evaluation = generate_family_examples(11003, 128, (8, 10), (72, 104), ("qaoa", "vqe", "arithmetic", "chemistry"))
    for split in (training, validation, evaluation): label_utilities(split)
    rows = {}; models = {}
    for kind in ("dag", "canonical_grid"):
        models[kind], training_result = train_ranker(kind, training, validation, 111, updates)
        rows[kind] = {"training": training_result, "evaluation": evaluate(models[kind], evaluation)}
    for kind in ("critical_path", "source_depth", "first", "random"):
        rows[kind] = {"evaluation": evaluate_scores(evaluation, heuristic_scores(evaluation, kind, 112))}
    strongest = max(rows[k]["evaluation"]["cx_recovery"] for k in ("critical_path", "source_depth", "first", "random"))
    passed = rows["dag"]["evaluation"]["mean_cx_reduction"] > rows["canonical_grid"]["evaluation"]["mean_cx_reduction"] and rows["dag"]["evaluation"]["cx_recovery"] >= strongest
    result = {"protocol": "frozen routed-rewrite development protocol", "updates": updates, "splits": {"train": 240, "validation": 72, "evaluation": 128}, "methods": rows, "development_gate_passed": passed}
    (output / "development_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"methods": {k: v["evaluation"] for k, v in rows.items()}, "development_gate_passed": passed}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=Path("experiments/routed_rewrite_development_20261101")); parser.add_argument("--updates", type=int, default=1500)
    args = parser.parse_args(); run_development(args.output, args.updates)


if __name__ == "__main__": main()
