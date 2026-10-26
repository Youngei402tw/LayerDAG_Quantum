"""Bounded 2-qubit LayerDAG-derived tiny learning experiment.

The model predicts a complete legal next dependency frontier.  Rotations use a
periodic categorical bin and a learned residual; no generated-angle optimizer is
used.  This is deliberately small and CPU-oriented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import Op, resources, simulate, wire_dag

NQ = 2
NBINS = 16
BIN_WIDTH = 2 * math.pi / NBINS
MAX_LAYERS = 6
DIFFUSION_STEPS = 8
STOP = 0


@dataclass(frozen=True)
class LayerAction:
    kind: str
    left_gate: str | None = None
    left_bin: int | None = None
    right_gate: str | None = None
    right_bin: int | None = None
    control: int | None = None


def action_catalog() -> list[LayerAction]:
    out = [LayerAction("stop"), LayerAction("cx", control=0), LayerAction("cx", control=1)]
    rotations = [(g, b) for g in ("ry", "rz") for b in range(NBINS)]
    for q in range(2):
        for gate, angle_bin in rotations:
            if q == 0:
                out.append(LayerAction("rot", gate, angle_bin))
            else:
                out.append(LayerAction("rot", right_gate=gate, right_bin=angle_bin))
    for gate0, bin0 in rotations:
        for gate1, bin1 in rotations:
            out.append(LayerAction("rot", gate0, bin0, gate1, bin1))
    return out


CATALOG = action_catalog()
ACTION_TO_ID = {action: i for i, action in enumerate(CATALOG)}
MASK = len(CATALOG)


def wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def angle_bin(theta: float) -> tuple[int, float]:
    theta = wrap_angle(theta)
    index = min(NBINS - 1, int((theta + math.pi) / BIN_WIDTH))
    center = -math.pi + (index + 0.5) * BIN_WIDTH
    return index, wrap_angle(theta - center)


def bin_angle(index: int, residual: float) -> float:
    center = -math.pi + (index + 0.5) * BIN_WIDTH
    return wrap_angle(center + float(np.clip(residual, -BIN_WIDTH / 2, BIN_WIDTH / 2)))


def encode_layer(ops: list[Op]) -> tuple[int, tuple[float, float]]:
    if not ops:
        return STOP, (0.0, 0.0)
    if len(ops) == 1 and ops[0].gate == "cx":
        action = LayerAction("cx", control=ops[0].wires[0])
        return ACTION_TO_ID[action], (0.0, 0.0)
    slots: list[tuple[str | None, int | None, float]] = [(None, None, 0.0), (None, None, 0.0)]
    for op in ops:
        if op.gate not in ("ry", "rz") or len(op.wires) != 1:
            raise ValueError("tiny action vocabulary supports RY/RZ/CX")
        q = op.wires[0]
        if slots[q][0] is not None:
            raise ValueError("frontier operations must use disjoint wires")
        b, residual = angle_bin(op.params[0])
        slots[q] = (op.gate, b, residual)
    action = LayerAction("rot", slots[0][0], slots[0][1], slots[1][0], slots[1][1])
    return ACTION_TO_ID[action], (slots[0][2], slots[1][2])


def decode_layer(action_id: int, residual: tuple[float, float] | np.ndarray) -> list[Op]:
    action = CATALOG[action_id]
    if action.kind == "stop":
        return []
    if action.kind == "cx":
        return [Op("cx", (action.control, 1 - action.control))]
    out: list[Op] = []
    if action.left_gate is not None:
        out.append(Op(action.left_gate, (0,), (bin_angle(action.left_bin, residual[0]),)))
    if action.right_gate is not None:
        out.append(Op(action.right_gate, (1,), (bin_angle(action.right_bin, residual[1]),)))
    return out


def flatten(layers: list[list[Op]]) -> list[Op]:
    return [op for layer in layers for op in layer]


def canonical_layers(ops: list[Op]) -> list[list[Op]]:
    """Schedule an ordered circuit at each operation's earliest wire depth."""
    frontier = [0] * NQ
    scheduled: list[tuple[int, Op]] = []
    for op in ops:
        depth = 1 + max(frontier[q] for q in op.wires)
        for q in op.wires:
            frontier[q] = depth
        scheduled.append((depth, op))
    layers: list[list[Op]] = [[] for _ in range(max(frontier, default=0))]
    for depth, op in scheduled:
        layers[depth - 1].append(op)
    for layer in layers:
        layer.sort(key=lambda op: op.wires)
    return layers


def density_features(state: np.ndarray) -> np.ndarray:
    rho = np.outer(state, state.conj())
    return np.concatenate([rho.real.ravel(), rho.imag.ravel()]).astype(np.float32)


def layer_cost(action_id: int) -> tuple[int, int, int]:
    action = CATALOG[action_id]
    if action.kind == "stop":
        return 0, 0, 0
    if action.kind == "cx":
        return 1, 1, 1
    return int(action.left_gate is not None) + int(action.right_gate is not None), 1, 0


def legal_mask(remaining: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    gates, depth, cx = remaining
    mask = torch.zeros(len(CATALOG), dtype=torch.bool, device=device)
    for i in range(len(CATALOG)):
        g, d, c = layer_cost(i)
        mask[i] = g <= gates and d <= depth and c <= cx
    mask[STOP] = True
    return mask


def random_layer(rng: np.random.Generator) -> list[Op]:
    if rng.random() < 0.25:
        control = int(rng.integers(2))
        return [Op("cx", (control, 1 - control))]
    wires = [int(rng.integers(2))] if rng.random() < 0.45 else [0, 1]
    return [Op("ry" if rng.random() < 0.55 else "rz", (q,), (float(rng.uniform(-math.pi, math.pi)),)) for q in wires]


def generate_targets(path: Path, seed: int = 20261021, count: int = 32, exclude_payload: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    records = []
    excluded = [] if exclude_payload is None else [density_features(target_array(record)) for record in exclude_payload["records"]]
    density_vectors: list[np.ndarray] = list(excluded)
    attempts = 0
    while len(records) < count and attempts < 10000:
        attempts += 1
        length = int(rng.integers(3, MAX_LAYERS + 1))
        proposed_layers = [random_layer(rng) for _ in range(length)]
        if not any(op.gate == "cx" for op in flatten(proposed_layers)):
            control = int(rng.integers(2))
            proposed_layers[int(rng.integers(length))] = [Op("cx", (control, 1 - control))]
        layers = canonical_layers(flatten(proposed_layers))
        if not 3 <= len(layers) <= MAX_LAYERS:
            continue
        target = simulate(NQ, flatten(layers))
        density = density_features(target)
        if any(np.linalg.norm(density - prior) <= 1e-10 for prior in density_vectors):
            continue
        density_vectors.append(density)
        encoded = [encode_layer(layer) for layer in layers]
        actual = resources(NQ, flatten(layers))
        if actual["depth"] != len(layers):
            raise AssertionError("canonical layer/depth mismatch")
        record = {
            "request_id": f"tiny_{len(records):02d}",
            "target_real": target.real.tolist(),
            "target_imag": target.imag.tolist(),
            "layers": [[asdict(op) for op in layer] for layer in layers],
            "action_ids": [item[0] for item in encoded],
            "residuals": [item[1] for item in encoded],
            "budgets": [actual["unitary_gate_count"], actual["depth"], actual["gate_counts"].get("cx", 0)],
            "witness_fidelity": float(abs(np.vdot(target, simulate(NQ, flatten(layers)))) ** 2),
        }
        records.append(record)
    if len(records) != count:
        raise RuntimeError("could not generate enough unique targets")
    payload = {
        "schema": 1,
        "seed": seed,
        "grouping": "phase-invariant density feature L2 distance > 1e-10",
        "excluded_target_groups": len(excluded),
        "basis_order": "qiskit_little_endian",
        "gate_set": ["ry", "rz", "cx"],
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload


def load_targets(path: Path) -> dict:
    return json.loads(path.read_text())


def record_layers(record: dict) -> list[list[Op]]:
    return [[Op(op["gate"], tuple(op["wires"]), tuple(op["params"])) for op in layer] for layer in record["layers"]]


def target_array(record: dict) -> np.ndarray:
    return np.array(record["target_real"]) + 1j * np.array(record["target_imag"])


class NativeMPNN(nn.Module):
    """Two-block bidirectional index-add message passing with labelled edges."""
    def __init__(self, width: int = 64):
        super().__init__()
        self.node_in = nn.Linear(12, width)
        self.forward_layers = nn.ModuleList([nn.Linear(width + 2, width) for _ in range(2)])
        self.backward_layers = nn.ModuleList([nn.Linear(width + 2, width) for _ in range(2)])
        self.self_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.out = nn.Linear(width * 3, width)

    @staticmethod
    def tensors(layers: list[list[Op]], device: torch.device):
        ops = flatten(layers)
        if not ops:
            return torch.zeros((0, 12), device=device), torch.zeros((2, 0), dtype=torch.long, device=device), torch.zeros((0, 2), device=device)
        nodes = []
        depths = []
        for depth, layer in enumerate(layers, 1):
            for op in layer:
                gate = [op.gate == "ry", op.gate == "rz", op.gate == "cx"]
                wire = [float(q in op.wires) for q in range(2)]
                roles = [float(op.gate == "cx" and op.wires[0] == q) for q in range(2)]
                theta = op.params[0] if op.params else 0.0
                nodes.append(gate + wire + roles + [math.sin(theta), math.cos(theta), depth / MAX_LAYERS, len(op.wires) / 2, 1.0])
                depths.append(depth)
        dag = wire_dag(NQ, ops)
        src, dst, edge_attr = [], [], []
        for node in dag:
            for incoming in node["incoming"]:
                if incoming["source"] is not None:
                    src.append(incoming["source"])
                    dst.append(node["id"])
                    edge_attr.append([incoming["wire"] == 0, incoming["wire"] == 1])
        edge = torch.tensor([src, dst], dtype=torch.long, device=device) if src else torch.zeros((2, 0), dtype=torch.long, device=device)
        attr = torch.tensor(edge_attr, dtype=torch.float32, device=device) if src else torch.zeros((0, 2), device=device)
        return torch.tensor(nodes, dtype=torch.float32, device=device), edge, attr

    def forward(self, batch_layers: list[list[list[Op]]]) -> torch.Tensor:
        outputs = []
        device = self.node_in.weight.device
        for layers in batch_layers:
            x, edge, attr = self.tensors(layers, device)
            if len(x) == 0:
                outputs.append(torch.zeros(self.out.out_features, device=device))
                continue
            h = F.gelu(self.node_in(x))
            states = [h]
            for fwd, bwd, self_layer in zip(self.forward_layers, self.backward_layers, self.self_layers):
                aggregate_f = torch.zeros_like(h)
                aggregate_b = torch.zeros_like(h)
                if edge.shape[1]:
                    messages_f = fwd(torch.cat([h[edge[0]], attr], dim=1))
                    messages_b = bwd(torch.cat([h[edge[1]], attr], dim=1))
                    aggregate_f.index_add_(0, edge[1], messages_f)
                    aggregate_b.index_add_(0, edge[0], messages_b)
                h = F.gelu(self_layer(h) + aggregate_f + aggregate_b)
                states.append(h)
            pooled = torch.cat([state.mean(0) for state in states])
            outputs.append(self.out(pooled))
        return torch.stack(outputs)


class GridEncoder(nn.Module):
    def __init__(self, width: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(len(CATALOG), width)
        self.gru = nn.GRU(width, width, batch_first=True, num_layers=2)

    def forward(self, batch_layers: list[list[list[Op]]]) -> torch.Tensor:
        device = self.embedding.weight.device
        sequences = []
        lengths = []
        for layers in batch_layers:
            ids = [encode_layer(layer)[0] for layer in layers]
            lengths.append(max(1, len(ids)))
            sequences.append(ids or [STOP])
        max_len = max(lengths)
        padded = torch.full((len(sequences), max_len), STOP, dtype=torch.long, device=device)
        for i, ids in enumerate(sequences):
            padded[i, :len(ids)] = torch.tensor(ids, device=device)
        encoded, _ = self.gru(self.embedding(padded))
        return torch.stack([encoded[i, length - 1] if sequences[i] != [STOP] else encoded[i, 0] * 0 for i, length in enumerate(lengths)])


class FrontierModel(nn.Module):
    def __init__(self, representation: str, width: int = 64):
        super().__init__()
        self.representation = representation
        self.prefix_encoder = NativeMPNN(width) if representation == "dag" else GridEncoder(width)
        self.physics = nn.Sequential(nn.Linear(67, width), nn.GELU(), nn.Linear(width, width))
        self.noisy_embedding = nn.Embedding(len(CATALOG) + 1, 32)
        self.time_embedding = nn.Embedding(DIFFUSION_STEPS + 1, 16)
        self.body = nn.Sequential(nn.Linear(width * 2 + 48, width * 2), nn.GELU(), nn.Linear(width * 2, width), nn.GELU())
        self.action_head = nn.Linear(width, len(CATALOG))
        self.residual_head = nn.Linear(width, 2)

    def forward(self, layers, target_features, prefix_features, remaining, noisy, step):
        graph = self.prefix_encoder(layers)
        physics = self.physics(torch.cat([target_features, prefix_features, remaining], dim=1))
        h = self.body(torch.cat([graph, physics, self.noisy_embedding(noisy), self.time_embedding(step)], dim=1))
        logits = self.action_head(h)
        residual = torch.tanh(self.residual_head(h)) * (BIN_WIDTH / 2)
        return logits, residual


def make_examples(payload: dict) -> list[dict]:
    examples = []
    for record in payload["records"]:
        layers = record_layers(record)
        target = target_array(record)
        budgets = tuple(record["budgets"])
        for prefix_length in range(len(layers) + 1):
            prefix = layers[:prefix_length]
            used = resources(NQ, flatten(prefix))
            remaining = (budgets[0] - used["unitary_gate_count"], budgets[1] - used["depth"], budgets[2] - used["gate_counts"].get("cx", 0))
            if prefix_length == len(layers):
                action_id, residual = STOP, (0.0, 0.0)
            else:
                action_id, residual = encode_layer(layers[prefix_length])
            examples.append({"layers": prefix, "target": target, "remaining": remaining, "action": action_id, "residual": residual})
    return examples


def batch_tensors(model: FrontierModel, batch: list[dict], rng: np.random.Generator, train: bool = True):
    device = next(model.parameters()).device
    target = torch.tensor(np.stack([density_features(x["target"]) for x in batch]), device=device)
    prefix_states = [simulate(NQ, flatten(x["layers"])) for x in batch]
    prefix = torch.tensor(np.stack([density_features(state) for state in prefix_states]), device=device)
    remaining = torch.tensor(np.array([x["remaining"] for x in batch], np.float32) / np.array([12, MAX_LAYERS, MAX_LAYERS], np.float32), device=device)
    clean = torch.tensor([x["action"] for x in batch], dtype=torch.long, device=device)
    residual = torch.tensor([x["residual"] for x in batch], dtype=torch.float32, device=device)
    if train:
        steps_np = rng.integers(1, DIFFUSION_STEPS + 1, size=len(batch))
        corrupt = rng.random(len(batch)) < steps_np / DIFFUSION_STEPS
    else:
        steps_np = np.ones(len(batch), dtype=int)
        corrupt = np.zeros(len(batch), dtype=bool)
    noisy = clean.clone()
    noisy[torch.tensor(corrupt, device=device)] = MASK
    steps = torch.tensor(steps_np, dtype=torch.long, device=device)
    return target, prefix, remaining, clean, residual, noisy, steps


def residual_mask(actions: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros((len(actions), 2), dtype=torch.float32, device=actions.device)
    for i, action_id in enumerate(actions.tolist()):
        action = CATALOG[action_id]
        if action.kind == "rot":
            mask[i, 0] = action.left_gate is not None
            mask[i, 1] = action.right_gate is not None
    return mask


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_model(representation: str, payload: dict, output: Path, seed: int, max_updates: int = 20000, cap_seconds: int = 1800, width: int = 64):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)
    model = FrontierModel(representation, width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    examples = make_examples(payload)
    rng = np.random.default_rng(seed)
    started = time.monotonic()
    best_loss = float("inf")
    best_state = None
    history = []
    batch_size = min(64, len(examples))
    for update in range(1, max_updates + 1):
        if time.monotonic() - started >= cap_seconds:
            status = "training_cap_reached"
            break
        selection = rng.integers(0, len(examples), size=batch_size)
        batch = [examples[int(i)] for i in selection]
        tensors = batch_tensors(model, batch, rng, train=True)
        logits, predicted_residual = model([x["layers"] for x in batch], tensors[0], tensors[1], tensors[2], tensors[5], tensors[6])
        action_loss = F.cross_entropy(logits, tensors[3])
        rmask = residual_mask(tensors[3])
        angle_loss = (((predicted_residual - tensors[4]) ** 2) * rmask).sum() / rmask.sum().clamp(min=1)
        loss = action_loss + 10.0 * angle_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % 100 == 0 or update == 1:
            full_losses = []
            model.eval()
            with torch.no_grad():
                for start in range(0, len(examples), 64):
                    eval_batch = examples[start:start + 64]
                    et = batch_tensors(model, eval_batch, rng, train=False)
                    el, er = model([x["layers"] for x in eval_batch], et[0], et[1], et[2], et[3], torch.zeros(len(eval_batch), dtype=torch.long))
                    em = residual_mask(et[3])
                    full_losses.append((F.cross_entropy(el, et[3]) + 10.0 * ((((er - et[4]) ** 2) * em).sum() / em.sum().clamp(min=1))).item())
            validation_loss = float(np.mean(full_losses))
            history.append({"update": update, "loss": float(loss.item()), "teacher_forced_loss": validation_loss, "elapsed_seconds": time.monotonic() - started})
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
            if validation_loss < 0.015 and update >= 500:
                status = "loss_gate_reached"
                break
    else:
        status = "update_cap_reached"
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.monotonic() - started
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": rng.bit_generator.state,
        "representation": representation,
        "width": width,
        "updates": update,
        "elapsed_seconds": elapsed,
        "best_teacher_forced_loss": best_loss,
        "status": status,
        "parameters": parameter_count(model),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    return model, checkpoint, history


def remaining_budget(budgets: tuple[int, int, int], layers: list[list[Op]]) -> tuple[int, int, int]:
    used = resources(NQ, flatten(layers))
    return (budgets[0] - used["unitary_gate_count"], budgets[1] - used["depth"], budgets[2] - used["gate_counts"].get("cx", 0))


@torch.no_grad()
def sample_frontier(model: FrontierModel, condition_target: np.ndarray | None, layers: list[list[Op]], remaining: tuple[int, int, int], rng: np.random.Generator):
    device = next(model.parameters()).device
    condition = np.zeros(32, np.float32) if condition_target is None else density_features(condition_target)
    target_f = torch.tensor(condition[None, :], device=device)
    prefix_f = torch.tensor(density_features(simulate(NQ, flatten(layers)))[None, :], device=device)
    budget_f = torch.tensor((np.array(remaining, np.float32) / np.array([12, MAX_LAYERS, MAX_LAYERS], np.float32))[None, :], device=device)
    noisy_id = MASK
    chosen = STOP
    for step in range(DIFFUSION_STEPS, 0, -1):
        noisy = torch.tensor([noisy_id], dtype=torch.long, device=device)
        step_t = torch.tensor([step], dtype=torch.long, device=device)
        logits, _ = model([layers], target_f, prefix_f, budget_f, noisy, step_t)
        allowed = legal_mask(remaining, device)
        logits[0, ~allowed] = -torch.inf
        probabilities = torch.softmax(logits[0], dim=0).cpu().numpy()
        chosen = int(rng.choice(len(CATALOG), p=probabilities))
        if rng.random() <= 1.0 / step:
            noisy_id = chosen
    clean = torch.tensor([chosen], dtype=torch.long, device=device)
    _, residual = model([layers], target_f, prefix_f, budget_f, clean, torch.zeros(1, dtype=torch.long, device=device))
    return chosen, residual[0].cpu().numpy()


def sample_circuit(model: FrontierModel, target: np.ndarray, budgets: tuple[int, int, int], prefix: list[list[Op]], rng: np.random.Generator, condition_target: np.ndarray | None | str = "correct"):
    layers = [list(layer) for layer in prefix]
    model_calls = 0
    condition = target if isinstance(condition_target, str) else condition_target
    while True:
        remaining = remaining_budget(budgets, layers)
        if remaining[1] <= 0 or remaining[0] <= 0:
            break
        action, residual = sample_frontier(model, condition, layers, remaining, rng)
        model_calls += DIFFUSION_STEPS + 1
        if action == STOP:
            break
        layer = decode_layer(action, residual)
        if not legal_mask(remaining, torch.device("cpu"))[action]:
            raise AssertionError("hard-mask failure")
        layers.append(layer)
    return layers, model_calls


def damaged_prefix(target: np.ndarray, witness_layers: list[list[Op]], budgets: tuple[int, int, int] | None = None) -> list[list[Op]]:
    """Choose the longest strict prefix that cannot already meet F>=0.99.

    If every strict witness prefix is already close enough (a redundant-tail
    witness), perturb the first rotation without changing its resource tuple.
    This keeps the repair task nontrivial and within the original maxima.
    """
    for length in range(len(witness_layers) - 1, -1, -1):
        prefix = witness_layers[:length]
        fidelity = float(abs(np.vdot(target, simulate(NQ, flatten(prefix)))) ** 2)
        if fidelity < 0.99:
            return prefix
    if budgets is None:
        actual = resources(NQ, flatten(witness_layers))
        budgets = (actual["unitary_gate_count"], actual["depth"], actual["gate_counts"].get("cx", 0))
    candidates = []
    for action_id in range(1, len(CATALOG)):
        cost = layer_cost(action_id)
        if cost[0] >= budgets[0] or cost[1] >= budgets[1] or cost[2] > budgets[2]:
            continue
        for residual in ((0.0, 0.0), (-BIN_WIDTH / 2, BIN_WIDTH / 2), (BIN_WIDTH / 2, -BIN_WIDTH / 2)):
            prefix = [decode_layer(action_id, residual)]
            fidelity = float(abs(np.vdot(target, simulate(NQ, flatten(prefix)))) ** 2)
            if fidelity < 0.99:
                candidates.append((fidelity, prefix))
    if not candidates:
        raise RuntimeError("could not construct a nonqualifying damaged seed")
    return max(candidates, key=lambda item: item[0])[1]


def evaluate_model(model: FrontierModel, payload: dict, samples: int = 32) -> dict:
    model.eval()
    rows = []
    for mode in ("scratch", "suffix_repair"):
        for record in payload["records"]:
            target = target_array(record)
            witness_layers = record_layers(record)
            budgets = tuple(record["budgets"])
            if mode == "scratch":
                prefix = []
            else:
                prefix = damaged_prefix(target, witness_layers, budgets)
            prefix_fidelity = float(abs(np.vdot(target, simulate(NQ, flatten(prefix)))) ** 2)
            candidates = []
            for k in range(samples):
                digest = hashlib.sha256(f"{record['request_id']}:{mode}:{k}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                layers, model_calls = sample_circuit(model, target, budgets, prefix, rng)
                ops = flatten(layers)
                actual = resources(NQ, ops)
                fidelity = float(abs(np.vdot(target, simulate(NQ, ops))) ** 2)
                feasible = actual["unitary_gate_count"] <= budgets[0] and actual["depth"] <= budgets[1] and actual["gate_counts"].get("cx", 0) <= budgets[2]
                candidates.append({"sample": k, "layers": layers, "fidelity": fidelity, "feasible": feasible, "resources": actual, "model_calls": model_calls})
            best = max(candidates, key=lambda x: x["fidelity"] if x["feasible"] else -1)
            one = candidates[0]
            rows.append({
                "request_id": record["request_id"], "mode": mode, "prefix_fidelity": prefix_fidelity,
                "k1_fidelity": one["fidelity"], "k1_success": one["feasible"] and one["fidelity"] >= 0.99,
                "k32_fidelity": best["fidelity"], "k32_success": best["feasible"] and best["fidelity"] >= 0.99,
                "best_resources": best["resources"], "best_layers": [[asdict(op) for op in layer] for layer in best["layers"]],
                "all_candidates_feasible": all(x["feasible"] for x in candidates),
                "model_calls": sum(x["model_calls"] for x in candidates),
                "unchanged_prefix_qualifies": prefix_fidelity >= 0.99,
            })
    summary = {}
    for mode in ("scratch", "suffix_repair"):
        group = [row for row in rows if row["mode"] == mode]
        summary[mode] = {
            "targets": len(group),
            "k1_success": sum(row["k1_success"] for row in group),
            "k32_success": sum(row["k32_success"] for row in group),
            "mean_k1_fidelity": float(np.mean([row["k1_fidelity"] for row in group])),
            "mean_k32_fidelity": float(np.mean([row["k32_fidelity"] for row in group])),
            "all_candidates_feasible": all(row["all_candidates_feasible"] for row in group),
            "unchanged_prefix_qualified": sum(row["unchanged_prefix_qualifies"] for row in group),
        }
    return {"summary": summary, "rows": rows}


def bootstrap_interval(values: np.ndarray, seed: int = 20261021, draws: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for i in range(draws):
        means[i] = np.mean(values[rng.integers(0, len(values), len(values))])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def conditioning_evaluation(model: FrontierModel, payload: dict, samples: int = 8) -> dict:
    """Paired held-out correct/permuted/null target diagnostic."""
    model.eval()
    records = payload["records"]
    rows = []
    for index, record in enumerate(records):
        requested = target_array(record)
        permuted = target_array(records[(index + 1) % len(records)])
        budgets = tuple(record["budgets"])
        condition_map = {"correct": requested, "permuted": permuted, "null": None}
        scores = {}
        for label, condition in condition_map.items():
            fidelities = []
            for k in range(samples):
                digest = hashlib.sha256(f"conditioning:{record['request_id']}:{k}".encode()).digest()
                rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
                layers, _ = sample_circuit(model, requested, budgets, [], rng, condition_target=condition)
                fidelities.append(float(abs(np.vdot(requested, simulate(NQ, flatten(layers)))) ** 2))
            scores[label] = max(fidelities)
        rows.append({"request_id": record["request_id"], **scores})
    correct = np.array([row["correct"] for row in rows])
    permuted = np.array([row["permuted"] for row in rows])
    null = np.array([row["null"] for row in rows])
    delta = correct - permuted
    interval = bootstrap_interval(delta)
    return {
        "targets": len(rows), "samples_per_condition": samples,
        "mean_correct_fidelity": float(correct.mean()),
        "mean_permuted_fidelity": float(permuted.mean()),
        "mean_null_fidelity": float(null.mean()),
        "paired_correct_minus_permuted": float(delta.mean()),
        "bootstrap_95_interval": interval,
        "condition_use_gate_passed": interval[0] > 0,
        "rows": rows,
    }


def dense_message_oracle(h: torch.Tensor, edge: torch.Tensor, attr: torch.Tensor, layer: nn.Linear, reverse: bool = False):
    out = torch.zeros((len(h), layer.out_features), dtype=h.dtype)
    for e in range(edge.shape[1]):
        source, target = (edge[1, e], edge[0, e]) if reverse else (edge[0, e], edge[1, e])
        out[target] += layer(torch.cat([h[source], attr[e]]))
    return out


def run(output_dir: Path, representation: str, seed: int, max_updates: int, cap_seconds: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "tiny_targets.json"
    payload = load_targets(data_path) if data_path.exists() else generate_targets(data_path)
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    started = time.monotonic()
    model, checkpoint, history = train_model(representation, payload, output_dir / f"{representation}_best.pt", seed, max_updates, cap_seconds)
    evaluation = evaluate_model(model, payload)
    result = {
        "representation": representation,
        "seed": seed,
        "data_sha256": data_hash,
        "checkpoint": f"{representation}_best.pt",
        "training": {k: v for k, v in checkpoint.items() if k not in ("state_dict", "optimizer", "torch_rng", "numpy_rng")},
        "history": history,
        "evaluation": evaluation,
        "total_wall_seconds": time.monotonic() - started,
        "learning_gate_passed": evaluation["summary"]["suffix_repair"]["k32_success"] >= 29,
        "guidance": "off",
        "angle_refinement": "off",
        "final_test_claim": False,
    }
    (output_dir / f"{representation}_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"representation": representation, "training": result["training"], "evaluation": evaluation["summary"], "learning_gate_passed": result["learning_gate_passed"], "total_wall_seconds": result["total_wall_seconds"]}, indent=2))
    return result


def replay_checkpoint(output_dir: Path, representation: str) -> dict:
    data_path = output_dir / "tiny_targets.json"
    payload = load_targets(data_path)
    checkpoint_path = output_dir / f"{representation}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FrontierModel(representation, checkpoint["width"])
    model.load_state_dict(checkpoint["state_dict"])
    started = time.monotonic()
    evaluation = evaluate_model(model, payload)
    result_path = output_dir / f"{representation}_result.json"
    prior = json.loads(result_path.read_text()) if result_path.exists() else {}
    prior["evaluation"] = evaluation
    prior["learning_gate_passed"] = evaluation["summary"]["suffix_repair"]["k32_success"] >= 29
    prior["evaluation_replay_seconds"] = time.monotonic() - started
    prior["damaged_seed_protocol"] = "longest strict witness prefix with unchanged fidelity below 0.99; deterministic same-cost perturbation fallback"
    result_path.write_text(json.dumps(prior, indent=2))
    print(json.dumps({"representation": representation, "evaluation": evaluation["summary"], "learning_gate_passed": prior["learning_gate_passed"], "evaluation_replay_seconds": prior["evaluation_replay_seconds"]}, indent=2))
    return prior


def evaluate_conditioning(output_dir: Path, representation: str) -> dict:
    validation_path = output_dir / "conditioning_validation_targets.json"
    payload = load_targets(validation_path) if validation_path.exists() else generate_targets(validation_path, seed=20261022, count=32)
    checkpoint = torch.load(output_dir / f"{representation}_best.pt", map_location="cpu", weights_only=False)
    model = FrontierModel(representation, checkpoint["width"])
    model.load_state_dict(checkpoint["state_dict"])
    started = time.monotonic()
    result = conditioning_evaluation(model, payload)
    result["elapsed_seconds"] = time.monotonic() - started
    result["validation_data_sha256"] = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    (output_dir / f"{representation}_conditioning.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return result


def assert_disjoint_targets(train_payload: dict, validation_payload: dict, tolerance: float = 1e-10) -> None:
    train = [density_features(target_array(record)) for record in train_payload["records"]]
    validation = [density_features(target_array(record)) for record in validation_payload["records"]]
    minimum = min(float(np.linalg.norm(a - b)) for a in train for b in validation)
    if minimum <= tolerance:
        raise ValueError(f"cross-split near duplicate at distance {minimum}")


def run_development(output_dir: Path, representation: str, seed: int, max_updates: int = 10000, cap_seconds: int = 900, width: int = 64) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "development_train_targets.json"
    validation_path = output_dir / "development_validation_targets.json"
    train_payload = load_targets(train_path) if train_path.exists() else generate_targets(train_path, seed=20261023, count=128)
    validation_payload = load_targets(validation_path) if validation_path.exists() else generate_targets(validation_path, seed=20261022, count=32, exclude_payload=train_payload)
    assert_disjoint_targets(train_payload, validation_payload)
    started = time.monotonic()
    checkpoint_path = output_dir / f"{representation}_development_best.pt"
    model, checkpoint, history = train_model(representation, train_payload, checkpoint_path, seed, max_updates, cap_seconds, width)
    conditioning = conditioning_evaluation(model, validation_payload, samples=8)
    result = {
        "representation": representation,
        "seed": seed,
        "train_targets": 128,
        "validation_targets": 32,
        "train_data_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "validation_data_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "training": {k: v for k, v in checkpoint.items() if k not in ("state_dict", "optimizer", "torch_rng", "numpy_rng")},
        "history": history,
        "conditioning": conditioning,
        "total_wall_seconds": time.monotonic() - started,
        "final_test_claim": False,
    }
    (output_dir / f"{representation}_development_result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"representation": representation, "training": result["training"], "conditioning": {k: v for k, v in conditioning.items() if k != "rows"}, "total_wall_seconds": result["total_wall_seconds"]}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/tiny_learning_20261021"))
    parser.add_argument("--representation", choices=("dag", "grid"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-updates", type=int, default=20000)
    parser.add_argument("--cap-seconds", type=int, default=1800)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--conditioning-only", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    if args.development:
        run_development(args.output, args.representation, args.seed, min(args.max_updates, 10000), min(args.cap_seconds, 900), args.width)
    elif args.conditioning_only:
        evaluate_conditioning(args.output, args.representation)
    elif args.eval_only:
        replay_checkpoint(args.output, args.representation)
    else:
        run(args.output, args.representation, args.seed, args.max_updates, args.cap_seconds)


if __name__ == "__main__":
    main()
