"""Circuit-DAG representation benchmark for depth-critical gate prediction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import Op, wire_dag

GATES = ("h", "ry", "rz", "cx")
MAX_QUBITS = 12


@dataclass
class CircuitExample:
    n_qubits: int
    ops: list[Op]
    labels: np.ndarray


def random_ops(rng: np.random.Generator, n_qubits: int, length: int) -> list[Op]:
    ops: list[Op] = []
    for _ in range(length):
        if rng.random() < 0.35:
            control, target = rng.choice(n_qubits, size=2, replace=False).tolist()
            ops.append(Op("cx", (int(control), int(target))))
        else:
            wire = int(rng.integers(n_qubits))
            gate = ("h", "ry", "rz")[int(rng.integers(3))]
            params = () if gate == "h" else (float(rng.uniform(-math.pi, math.pi)),)
            ops.append(Op(gate, (wire,), params))
    return ops


def dependency_edges(n_qubits: int, ops: list[Op]) -> tuple[np.ndarray, np.ndarray]:
    src, dst, wires = [], [], []
    for node in wire_dag(n_qubits, ops):
        for incoming in node["incoming"]:
            if incoming["source"] is not None:
                src.append(incoming["source"])
                dst.append(node["id"])
                wires.append(incoming["wire"])
    return np.asarray([src, dst], np.int64), np.asarray(wires, np.int64)


def critical_labels(n_qubits: int, ops: list[Op]) -> np.ndarray:
    edge, _ = dependency_edges(n_qubits, ops)
    count = len(ops)
    predecessors = [[] for _ in range(count)]
    successors = [[] for _ in range(count)]
    for source, target in edge.T:
        predecessors[int(target)].append(int(source))
        successors[int(source)].append(int(target))
    forward = np.ones(count, np.int64)
    for node in range(count):
        if predecessors[node]:
            forward[node] = 1 + max(forward[parent] for parent in predecessors[node])
    backward = np.ones(count, np.int64)
    for node in range(count - 1, -1, -1):
        if successors[node]:
            backward[node] = 1 + max(backward[child] for child in successors[node])
    depth = int(forward.max())
    return (forward + backward - 1 == depth).astype(np.float32)


def random_topological_permutation(rng: np.random.Generator, n_qubits: int, ops: list[Op]) -> list[int]:
    edge, _ = dependency_edges(n_qubits, ops)
    successors = [[] for _ in ops]
    indegree = np.zeros(len(ops), np.int64)
    for source, target in edge.T:
        successors[int(source)].append(int(target))
        indegree[int(target)] += 1
    ready = [i for i, degree in enumerate(indegree) if degree == 0]
    order = []
    while ready:
        pick = int(rng.integers(len(ready)))
        node = ready.pop(pick)
        order.append(node)
        for child in successors[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != len(ops):
        raise AssertionError("dependency graph is cyclic")
    return order


def reorder_example(example: CircuitExample, seed: int) -> CircuitExample:
    order = random_topological_permutation(np.random.default_rng(seed), example.n_qubits, example.ops)
    return CircuitExample(example.n_qubits, [example.ops[i] for i in order], example.labels[order])


def canonical_order(example: CircuitExample) -> list[int]:
    edge, _ = dependency_edges(example.n_qubits, example.ops)
    predecessors = [[] for _ in example.ops]
    for source, target in edge.T:
        predecessors[int(target)].append(int(source))
    depth = np.ones(len(example.ops), np.int64)
    for node in range(len(example.ops)):
        if predecessors[node]:
            depth[node] = 1 + max(depth[parent] for parent in predecessors[node])
    return sorted(range(len(example.ops)), key=lambda i: (
        int(depth[i]), example.ops[i].wires, example.ops[i].gate, example.ops[i].params
    ))


def generate_examples(seed: int, count: int, qubits: tuple[int, int], lengths: tuple[int, int]) -> list[CircuitExample]:
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(count):
        n_qubits = int(rng.integers(qubits[0], qubits[1] + 1))
        length = int(rng.integers(lengths[0], lengths[1] + 1))
        ops = random_ops(rng, n_qubits, length)
        examples.append(CircuitExample(n_qubits, ops, critical_labels(n_qubits, ops)))
    return examples


def examples_hash(examples: list[CircuitExample]) -> str:
    rows = []
    for example in examples:
        rows.append({
            "n_qubits": example.n_qubits,
            "ops": [{"gate": op.gate, "wires": op.wires, "params": op.params} for op in example.ops],
            "labels": example.labels.tolist(),
        })
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def node_features(example: CircuitExample) -> np.ndarray:
    features = []
    for op in example.ops:
        gate = [float(op.gate == name) for name in GATES]
        membership = [float(q in op.wires) for q in range(MAX_QUBITS)]
        control = [float(op.gate == "cx" and op.wires[0] == q) for q in range(MAX_QUBITS)]
        target = [float(op.gate == "cx" and op.wires[1] == q) for q in range(MAX_QUBITS)]
        theta = op.params[0] if op.params else 0.0
        features.append(gate + membership + control + target + [math.sin(theta), math.cos(theta), len(op.wires) / 2])
    return np.asarray(features, np.float32)


class DagCriticalModel(nn.Module):
    def __init__(self, width: int = 64, layers: int = 6):
        super().__init__()
        input_size = len(GATES) + 3 * MAX_QUBITS + 3
        self.node_in = nn.Linear(input_size, width)
        self.forward_layers = nn.ModuleList([nn.Linear(width + MAX_QUBITS, width) for _ in range(layers)])
        self.backward_layers = nn.ModuleList([nn.Linear(width + MAX_QUBITS, width) for _ in range(layers)])
        self.self_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.out = nn.Linear(width, 1)

    def forward_one(self, example: CircuitExample) -> torch.Tensor:
        device = self.node_in.weight.device
        h = F.gelu(self.node_in(torch.tensor(node_features(example), device=device)))
        edge_np, wire_np = dependency_edges(example.n_qubits, example.ops)
        edge = torch.tensor(edge_np, dtype=torch.long, device=device)
        attr = F.one_hot(torch.tensor(wire_np, device=device), MAX_QUBITS).float()
        for fwd, bwd, self_layer, norm in zip(self.forward_layers, self.backward_layers, self.self_layers, self.norms):
            aggregate_f = torch.zeros_like(h)
            aggregate_b = torch.zeros_like(h)
            if edge.shape[1]:
                aggregate_f.index_add_(0, edge[1], fwd(torch.cat([h[edge[0]], attr], 1)))
                aggregate_b.index_add_(0, edge[0], bwd(torch.cat([h[edge[1]], attr], 1)))
            h = norm(h + F.gelu(self_layer(h) + aggregate_f + aggregate_b))
        return self.out(h).squeeze(1)

    def forward(self, examples: list[CircuitExample]) -> list[torch.Tensor]:
        return [self.forward_one(example) for example in examples]


class RecurrentDagCriticalModel(nn.Module):
    """Full-depth bidirectional recurrence over the circuit dependency DAG."""
    def __init__(self, width: int = 60):
        super().__init__()
        input_size = len(GATES) + 3 * MAX_QUBITS + 3
        self.node_in = nn.Linear(input_size, width)
        self.forward_cell = nn.GRUCell(width, width)
        self.backward_cell = nn.GRUCell(width, width)
        self.out = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))

    def forward_one(self, example: CircuitExample) -> torch.Tensor:
        device = self.node_in.weight.device
        x = F.gelu(self.node_in(torch.tensor(node_features(example), device=device)))
        edge, _ = dependency_edges(example.n_qubits, example.ops)
        predecessors = [[] for _ in example.ops]
        successors = [[] for _ in example.ops]
        for source, target in edge.T:
            predecessors[int(target)].append(int(source))
            successors[int(source)].append(int(target))
        zero = torch.zeros(self.node_in.out_features, device=device)
        forward: list[torch.Tensor] = []
        for node in range(len(example.ops)):
            aggregate = torch.stack([forward[parent] for parent in predecessors[node]]).amax(0) if predecessors[node] else zero
            forward.append(self.forward_cell(x[node], aggregate))
        backward: list[torch.Tensor | None] = [None] * len(example.ops)
        for node in range(len(example.ops) - 1, -1, -1):
            aggregate = torch.stack([backward[child] for child in successors[node]]).amax(0) if successors[node] else zero
            backward[node] = self.backward_cell(x[node], aggregate)
        return self.out(torch.cat([x, torch.stack(forward), torch.stack(backward)], 1)).squeeze(1)

    def forward(self, examples: list[CircuitExample]) -> list[torch.Tensor]:
        return [self.forward_one(example) for example in examples]


class SequenceCriticalModel(nn.Module):
    def __init__(self, width: int = 64, layers: int = 2):
        super().__init__()
        input_size = len(GATES) + 3 * MAX_QUBITS + 3
        self.node_in = nn.Linear(input_size, width)
        self.gru = nn.GRU(width, width // 2, num_layers=layers, bidirectional=True, batch_first=True)
        self.out = nn.Linear(width, 1)

    def forward(self, examples: list[CircuitExample]) -> list[torch.Tensor]:
        device = self.node_in.weight.device
        lengths = [len(example.ops) for example in examples]
        padded = torch.zeros((len(examples), max(lengths), self.node_in.in_features), device=device)
        for i, example in enumerate(examples):
            padded[i, :lengths[i]] = torch.tensor(node_features(example), device=device)
        packed = nn.utils.rnn.pack_padded_sequence(F.gelu(self.node_in(padded)), lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.gru(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)
        return [self.out(unpacked[i, :length]).squeeze(1) for i, length in enumerate(lengths)]


class CanonicalGridCriticalModel(SequenceCriticalModel):
    """Sequence control after deterministic earliest-layer canonicalization."""
    def forward(self, examples: list[CircuitExample]) -> list[torch.Tensor]:
        canonical = []
        orders = []
        for example in examples:
            order = canonical_order(example)
            orders.append(order)
            canonical.append(CircuitExample(example.n_qubits, [example.ops[i] for i in order], example.labels[order]))
        canonical_logits = super().forward(canonical)
        restored = []
        for logits, order in zip(canonical_logits, orders):
            output = torch.empty_like(logits)
            output[torch.tensor(order, device=logits.device)] = logits
            restored.append(output)
        return restored


class TransformerCriticalModel(nn.Module):
    """Strong flat-token control with bidirectional global attention."""
    def __init__(self, width: int = 64, layers: int = 2, max_length: int = 128):
        super().__init__()
        input_size = len(GATES) + 3 * MAX_QUBITS + 3
        self.node_in = nn.Linear(input_size, width)
        self.position = nn.Embedding(max_length, width)
        block = nn.TransformerEncoderLayer(width, 4, dim_feedforward=160, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.out = nn.Linear(width, 1)

    def forward(self, examples: list[CircuitExample]) -> list[torch.Tensor]:
        device = self.node_in.weight.device
        lengths = [len(example.ops) for example in examples]
        maximum = max(lengths)
        padded = torch.zeros((len(examples), maximum, self.node_in.in_features), device=device)
        padding_mask = torch.ones((len(examples), maximum), dtype=torch.bool, device=device)
        for i, example in enumerate(examples):
            padded[i, :lengths[i]] = torch.tensor(node_features(example), device=device)
            padding_mask[i, :lengths[i]] = False
        positions = self.position(torch.arange(maximum, device=device))[None, :, :]
        encoded = self.encoder(F.gelu(self.node_in(padded)) + positions, src_key_padding_mask=padding_mask)
        return [self.out(encoded[i, :length]).squeeze(1) for i, length in enumerate(lengths)]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) / (positive.sum() * negative.sum()))


@torch.no_grad()
def evaluate(model: nn.Module, examples: list[CircuitExample]) -> dict:
    model.eval()
    logits = []
    labels = []
    for start in range(0, len(examples), 32):
        batch = examples[start:start + 32]
        logits.extend(t.cpu().numpy() for t in model(batch))
        labels.extend(example.labels for example in batch)
    scores = 1 / (1 + np.exp(-np.concatenate(logits)))
    truth = np.concatenate(labels)
    predicted = scores >= 0.5
    tp = int(np.sum(predicted & (truth == 1)))
    fp = int(np.sum(predicted & (truth == 0)))
    fn = int(np.sum(~predicted & (truth == 1)))
    return {
        "nodes": len(truth),
        "positive_fraction": float(truth.mean()),
        "auc": auc_score(truth, scores),
        "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
        "accuracy": float(np.mean(predicted == truth)),
    }


def train(kind: str, train_examples: list[CircuitExample], validation: list[CircuitExample], seed: int,
          updates: int = 3000, width: int = 64) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    if kind == "dag":
        model = DagCriticalModel(width)
    elif kind == "dag_recurrent":
        model = RecurrentDagCriticalModel(width * 15 // 16)
    elif kind == "sequence":
        model = SequenceCriticalModel(width * 3 // 2)
    elif kind == "canonical_grid":
        model = CanonicalGridCriticalModel(width * 3 // 2)
    elif kind == "transformer":
        model = TransformerCriticalModel(width)
    else:
        raise ValueError(kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    positives = sum(float(example.labels.sum()) for example in train_examples)
    negatives = sum(len(example.labels) - float(example.labels.sum()) for example in train_examples)
    positive_weight = torch.tensor(negatives / positives)
    rng = np.random.default_rng(seed)
    best_auc = -1.0
    best_state = None
    history = []
    started = time.monotonic()
    for update in range(1, updates + 1):
        indices = rng.integers(0, len(train_examples), size=16)
        batch = [train_examples[int(i)] for i in indices]
        outputs = model(batch)
        loss = torch.stack([
            F.binary_cross_entropy_with_logits(logit, torch.tensor(example.labels), pos_weight=positive_weight)
            for logit, example in zip(outputs, batch)
        ]).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 250 == 0 or update == updates:
            metrics = evaluate(model, validation)
            history.append({"update": update, "loss": float(loss.item()), **metrics})
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {
        "kind": kind,
        "seed": seed,
        "updates": updates,
        "parameters": parameter_count(model),
        "width": (width * 3 // 2 if kind in ("sequence", "canonical_grid")
                  else width * 15 // 16 if kind == "dag_recurrent" else width),
        "train_seconds": time.monotonic() - started,
        "best_validation_auc": best_auc,
        "history": history,
    }


def run_development(output: Path, seed: int = 41, updates: int = 3000,
                    kinds: tuple[str, ...] = ("dag", "dag_recurrent", "sequence", "canonical_grid", "transformer")) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    training = generate_examples(4100, 1000, (4, 8), (24, 56))
    validation = generate_examples(4200, 256, (4, 8), (24, 56))
    reordered = [reorder_example(example, 4300 + i) for i, example in enumerate(validation)]
    larger = generate_examples(4400, 256, (9, 12), (64, 96))
    results = {}
    models = {}
    for kind in kinds:
        model, training_result = train(kind, training, validation, seed, updates)
        models[kind] = model
        results[kind] = {
            "training": training_result,
            "canonical": evaluate(model, validation),
            "reordered": evaluate(model, reordered),
            "larger": evaluate(model, larger),
        }
    with torch.no_grad() if "dag" in models else torch.enable_grad():
        invariance = []
        if "dag" in models:
            for original, changed in zip(validation[:64], reordered[:64]):
                original_logits = models["dag"]([original])[0]
                order = random_topological_permutation(np.random.default_rng(4300 + len(invariance)), original.n_qubits, original.ops)
                restored = torch.empty_like(original_logits)
                restored[torch.tensor(order)] = models["dag"]([changed])[0]
                invariance.append(float(torch.max(torch.abs(original_logits - restored))))
    result = {
        "protocol": "development only; 1000 train, 256 validation, 256 reordered, 256 larger OOD",
        "seed": seed,
        "results": results,
        "dag_max_reordering_logit_difference": max(invariance) if invariance else None,
    }
    (output / "development_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def run_final(output: Path, updates: int = 6000) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    training = generate_examples(6100, 1000, (4, 8), (24, 56))
    validation = generate_examples(6200, 256, (4, 8), (24, 56))
    test = generate_examples(6300, 512, (4, 8), (24, 56))
    reordered = [reorder_example(example, 6400 + i) for i, example in enumerate(test)]
    larger = generate_examples(6500, 512, (9, 12), (64, 96))
    seed_results = []
    for seed in (51, 52, 53):
        row = {"seed": seed, "models": {}}
        for kind in ("dag", "canonical_grid"):
            model, training_result = train(kind, training, validation, seed, updates)
            row["models"][kind] = {
                "training": training_result,
                "canonical_test": evaluate(model, test),
                "reordered_test": evaluate(model, reordered),
                "larger_test": evaluate(model, larger),
            }
        seed_results.append(row)
    aggregate = {}
    for kind in ("dag", "canonical_grid"):
        aggregate[kind] = {}
        for split in ("canonical_test", "reordered_test", "larger_test"):
            aggregate[kind][split] = {}
            for metric in ("auc", "f1", "accuracy"):
                values = np.array([row["models"][kind][split][metric] for row in seed_results])
                aggregate[kind][split][metric + "_mean"] = float(values.mean())
                aggregate[kind][split][metric + "_sample_std"] = float(values.std(ddof=1))
                aggregate[kind][split][metric + "_values"] = values.tolist()
    auc_differences = np.array([
        row["models"]["dag"]["larger_test"]["auc"] - row["models"]["canonical_grid"]["larger_test"]["auc"]
        for row in seed_results
    ])
    f1_difference = (aggregate["dag"]["larger_test"]["f1_mean"]
                     - aggregate["canonical_grid"]["larger_test"]["f1_mean"])
    result = {
        "protocol": "frozen fresh-data three-seed DAG versus canonical-grid size-transfer test",
        "updates_per_model": updates,
        "dataset_hashes": {
            "training": examples_hash(training),
            "validation": examples_hash(validation),
            "test": examples_hash(test),
            "reordered_test": examples_hash(reordered),
            "larger_test": examples_hash(larger),
        },
        "seed_results": seed_results,
        "aggregate": aggregate,
        "larger_auc_paired_differences": auc_differences.tolist(),
        "larger_auc_mean_difference": float(auc_differences.mean()),
        "larger_f1_mean_difference": float(f1_difference),
        "reordering_equivariance": "DAG architecture is permutation-equivariant over legal topological orders; unit-tested at atol=1e-6",
        "positive_claim_passed": bool(np.all(auc_differences > 0) and auc_differences.mean() >= 0.015 and f1_difference > 0),
    }
    (output / "final_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/schedule_dag_development_20261027"))
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--kinds", nargs="+", choices=("dag", "dag_recurrent", "sequence", "canonical_grid", "transformer"),
                        default=("dag", "dag_recurrent", "sequence", "canonical_grid", "transformer"))
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.final:
        run_final(args.output, args.updates)
    else:
        run_development(args.output, args.seed, args.updates, tuple(args.kinds))


if __name__ == "__main__":
    main()
