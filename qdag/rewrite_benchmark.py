"""Pilot learned ranking of fidelity-preserving gate deletions."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from .core import Op, simulate
from .schedule_benchmark import (
    CircuitExample, DagCriticalModel, CanonicalGridCriticalModel, evaluate, train,
)

FIDELITY_MIN = 0.999


def random_rewrite_ops(rng: np.random.Generator, n_qubits: int, length: int) -> list[Op]:
    ops: list[Op] = []
    for _ in range(length):
        draw = rng.random()
        if draw < 0.28:
            control, target = rng.choice(n_qubits, size=2, replace=False).tolist()
            ops.append(Op("cx", (int(control), int(target))))
        elif draw < 0.34:
            ops.append(Op("h", (int(rng.integers(n_qubits)),)))
        else:
            wire = int(rng.integers(n_qubits))
            gate = "ry" if rng.random() < 0.55 else "rz"
            angle = float(rng.uniform(-0.05, 0.05) if rng.random() < 0.28 else rng.uniform(-math.pi, math.pi))
            ops.append(Op(gate, (wire,), (angle,)))
    return ops


def deletion_fidelities(n_qubits: int, ops: list[Op]) -> np.ndarray:
    target = simulate(n_qubits, ops)
    scores = np.empty(len(ops), np.float64)
    for index in range(len(ops)):
        edited = simulate(n_qubits, ops[:index] + ops[index + 1:])
        scores[index] = float(abs(np.vdot(target, edited)) ** 2)
    return scores


def generate_rewrite_examples(seed: int, count: int, qubits: tuple[int, int], lengths: tuple[int, int]) -> list[CircuitExample]:
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(count):
        n_qubits = int(rng.integers(qubits[0], qubits[1] + 1))
        length = int(rng.integers(lengths[0], lengths[1] + 1))
        ops = random_rewrite_ops(rng, n_qubits, length)
        labels = (deletion_fidelities(n_qubits, ops) >= FIDELITY_MIN).astype(np.float32)
        examples.append(CircuitExample(n_qubits, ops, labels))
    return examples


@torch.no_grad()
def model_scores(model: torch.nn.Module, examples: list[CircuitExample]) -> list[np.ndarray]:
    model.eval()
    output = []
    for start in range(0, len(examples), 32):
        output.extend(logit.cpu().numpy() for logit in model(examples[start:start + 32]))
    return output


def heuristic_scores(example: CircuitExample) -> np.ndarray:
    scores = np.full(len(example.ops), -10.0)
    for index, op in enumerate(example.ops):
        if op.params:
            scores[index] = -abs(op.params[0])
        elif op.gate == "h":
            scores[index] = -4.0
        else:
            scores[index] = -5.0
    return scores


def ranking_metrics(examples: list[CircuitExample], rankings: list[np.ndarray],
                    fidelities: list[np.ndarray], budgets: tuple[int, ...] = (1, 4, 8)) -> dict:
    result = {"circuits": len(examples), "oracle_has_safe_deletion": float(np.mean([example.labels.any() for example in examples]))}
    for budget in budgets:
        successes = 0
        calls = []
        selected_fidelities = []
        for example, scores, fidelity in zip(examples, rankings, fidelities):
            order = np.argsort(-scores, kind="stable")[:budget]
            found = False
            for call, index in enumerate(order, 1):
                if fidelity[index] >= FIDELITY_MIN:
                    successes += 1
                    calls.append(call)
                    selected_fidelities.append(float(fidelity[index]))
                    found = True
                    break
            if not found:
                calls.append(len(order))
        result[f"budget_{budget}"] = {
            "success_rate": successes / len(examples),
            "mean_verification_calls": float(np.mean(calls)),
            "mean_selected_fidelity": float(np.mean(selected_fidelities)) if selected_fidelities else None,
        }
    return result


def run_pilot(output: Path, updates: int = 3000) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    training = generate_rewrite_examples(7100, 600, (4, 6), (20, 40))
    validation = generate_rewrite_examples(7200, 128, (4, 6), (20, 40))
    evaluation = generate_rewrite_examples(7300, 256, (7, 9), (40, 64))
    models = {}
    training_results = {}
    for kind in ("dag", "canonical_grid"):
        model, training_result = train(kind, training, validation, 71, updates)
        models[kind] = model
        training_results[kind] = training_result
    rankings = {kind: model_scores(model, evaluation) for kind, model in models.items()}
    rankings["angle_heuristic"] = [heuristic_scores(example) for example in evaluation]
    random_rng = np.random.default_rng(7400)
    rankings["random"] = [random_rng.random(len(example.ops)) for example in evaluation]
    evaluation_fidelities = [deletion_fidelities(example.n_qubits, example.ops) for example in evaluation]
    downstream = {kind: ranking_metrics(evaluation, scores, evaluation_fidelities) for kind, scores in rankings.items()}
    result = {
        "protocol": "development rewrite-ranking pilot; labels and acceptance from complete statevector simulation",
        "updates": updates,
        "train_circuits": len(training),
        "validation_circuits": len(validation),
        "size_shift_evaluation_circuits": len(evaluation),
        "training": training_results,
        "validation_node_metrics": {kind: evaluate(model, validation) for kind, model in models.items()},
        "evaluation_node_metrics": {kind: evaluate(model, evaluation) for kind, model in models.items()},
        "downstream": downstream,
        "pilot_gate_passed": bool(
            downstream["dag"]["budget_4"]["success_rate"] > downstream["canonical_grid"]["budget_4"]["success_rate"]
            and downstream["dag"]["budget_4"]["success_rate"] > downstream["angle_heuristic"]["budget_4"]["success_rate"]
        ),
    }
    (output / "pilot_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "validation_node_metrics": result["validation_node_metrics"],
        "evaluation_node_metrics": result["evaluation_node_metrics"],
        "downstream": downstream,
        "pilot_gate_passed": result["pilot_gate_passed"],
    }, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/rewrite_ranking_pilot_20261029"))
    parser.add_argument("--updates", type=int, default=3000)
    args = parser.parse_args()
    run_pilot(args.output, args.updates)


if __name__ == "__main__":
    main()
