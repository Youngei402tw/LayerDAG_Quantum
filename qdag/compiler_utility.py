"""Downstream compiler utility of learned depth-critical gate rankings."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .core import Op, resources
from .schedule_benchmark import (
    CircuitExample, critical_labels, generate_examples, random_ops, train,
)


@dataclass
class IdentityRewriteExample:
    circuit: CircuitExample
    candidates: list[int]


def generate_identity_rewrites(seed: int, count: int) -> list[IdentityRewriteExample]:
    rng = np.random.default_rng(seed)
    output = []
    for _ in range(count):
        n_qubits = int(rng.integers(9, 13))
        length = int(rng.integers(64, 97))
        ops = random_ops(rng, n_qubits, length)
        positions = sorted(rng.choice(len(ops) + 1, size=n_qubits, replace=False).tolist())
        insertions = [(int(position), Op("rz", (int(rng.integers(n_qubits)),), (0.0,))) for position in positions]
        candidates = []
        for offset, (position, op) in enumerate(insertions):
            adjusted = position + offset
            ops.insert(min(adjusted, len(ops)), op)
            candidates.append(min(adjusted, len(ops) - 1))
        candidates = sorted(set(candidates))
        circuit = CircuitExample(n_qubits, ops, critical_labels(n_qubits, ops))
        output.append(IdentityRewriteExample(circuit, candidates))
    return output


@torch.no_grad()
def score_candidates(model: torch.nn.Module, examples: list[IdentityRewriteExample]) -> list[np.ndarray]:
    model.eval()
    scores = []
    for start in range(0, len(examples), 16):
        batch = examples[start:start + 16]
        logits = model([example.circuit for example in batch])
        scores.extend(logit.cpu().numpy()[example.candidates] for logit, example in zip(logits, batch))
    return scores


def candidate_depth_reductions(example: IdentityRewriteExample) -> np.ndarray:
    circuit = example.circuit
    original = resources(circuit.n_qubits, circuit.ops)["depth"]
    reductions = []
    for index in example.candidates:
        edited = circuit.ops[:index] + circuit.ops[index + 1:]
        reductions.append(original - resources(circuit.n_qubits, edited)["depth"])
    return np.asarray(reductions, np.float64)


def utility_metrics(examples: list[IdentityRewriteExample], scores: list[np.ndarray]) -> dict:
    selected = []
    oracle = []
    exact = 0
    for example, candidate_scores in zip(examples, scores):
        reductions = candidate_depth_reductions(example)
        chosen = int(np.argmax(candidate_scores))
        selected.append(float(reductions[chosen]))
        oracle.append(float(reductions.max()))
        exact += reductions[chosen] == reductions.max()
    selected_a = np.asarray(selected)
    oracle_a = np.asarray(oracle)
    return {
        "circuits": len(examples),
        "mean_depth_reduction": float(selected_a.mean()),
        "any_depth_reduction_rate": float(np.mean(selected_a > 0)),
        "exact_oracle_choice_rate": exact / len(examples),
        "oracle_mean_depth_reduction": float(oracle_a.mean()),
        "oracle_reduction_recovered": float(selected_a.sum() / max(1.0, oracle_a.sum())),
    }


def run_pilot(output: Path, updates: int = 3000) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    training = generate_examples(8100, 1000, (4, 8), (24, 56))
    validation = generate_examples(8200, 256, (4, 8), (24, 56))
    evaluation = generate_identity_rewrites(8300, 256)
    models = {}
    training_results = {}
    for kind in ("dag", "canonical_grid"):
        model, training_result = train(kind, training, validation, 81, updates)
        models[kind] = model
        training_results[kind] = training_result
    rankings = {kind: score_candidates(model, evaluation) for kind, model in models.items()}
    rng = np.random.default_rng(8400)
    rankings["random"] = [rng.random(len(example.candidates)) for example in evaluation]
    utility = {kind: utility_metrics(evaluation, scores) for kind, scores in rankings.items()}
    result = {
        "protocol": "development pilot: rank exactly removable identities by realized dependency-depth reduction",
        "updates": updates,
        "training": training_results,
        "utility": utility,
        "pilot_gate_passed": bool(
            utility["dag"]["mean_depth_reduction"] > utility["canonical_grid"]["mean_depth_reduction"]
            and utility["dag"]["oracle_reduction_recovered"] > utility["canonical_grid"]["oracle_reduction_recovered"]
        ),
    }
    (output / "pilot_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"utility": utility, "pilot_gate_passed": result["pilot_gate_passed"]}, indent=2))
    return result


def run_final(output: Path, updates: int = 3000) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    training = generate_examples(9100, 1000, (4, 8), (24, 56))
    validation = generate_examples(9200, 256, (4, 8), (24, 56))
    evaluation = generate_identity_rewrites(9300, 512)
    seed_results = []
    for seed in (91, 92, 93):
        row = {"seed": seed, "models": {}}
        for kind in ("dag", "canonical_grid"):
            model, training_result = train(kind, training, validation, seed, updates)
            scores = score_candidates(model, evaluation)
            row["models"][kind] = {
                "training": training_result,
                "utility": utility_metrics(evaluation, scores),
            }
        seed_results.append(row)
    random_rng = np.random.default_rng(9400)
    random_scores = [random_rng.random(len(example.candidates)) for example in evaluation]
    random_utility = utility_metrics(evaluation, random_scores)
    metrics = ("mean_depth_reduction", "any_depth_reduction_rate", "exact_oracle_choice_rate", "oracle_reduction_recovered")
    aggregate = {}
    for kind in ("dag", "canonical_grid"):
        aggregate[kind] = {}
        for metric in metrics:
            values = np.array([row["models"][kind]["utility"][metric] for row in seed_results])
            aggregate[kind][metric + "_mean"] = float(values.mean())
            aggregate[kind][metric + "_sample_std"] = float(values.std(ddof=1))
            aggregate[kind][metric + "_values"] = values.tolist()
    recovery_differences = np.array([
        row["models"]["dag"]["utility"]["oracle_reduction_recovered"]
        - row["models"]["canonical_grid"]["utility"]["oracle_reduction_recovered"]
        for row in seed_results
    ])
    result = {
        "protocol": "frozen fresh-data three-seed safe-rewrite prioritization under circuit-size shift",
        "updates_per_model": updates,
        "seed_results": seed_results,
        "aggregate": aggregate,
        "random": random_utility,
        "recovery_paired_differences": recovery_differences.tolist(),
        "recovery_mean_difference": float(recovery_differences.mean()),
        "positive_claim_passed": bool(
            np.all(recovery_differences > 0)
            and recovery_differences.mean() >= 0.05
            and aggregate["dag"]["mean_depth_reduction_mean"] > aggregate["canonical_grid"]["mean_depth_reduction_mean"]
            and aggregate["dag"]["exact_oracle_choice_rate_mean"] > aggregate["canonical_grid"]["exact_oracle_choice_rate_mean"]
        ),
        "limitation": "rewrite candidates are injected exact RZ(0) identities; natural rewrite validation is not claimed",
    }
    (output / "final_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "aggregate": aggregate,
        "random": random_utility,
        "recovery_paired_differences": result["recovery_paired_differences"],
        "recovery_mean_difference": result["recovery_mean_difference"],
        "positive_claim_passed": result["positive_claim_passed"],
    }, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/critical_rewrite_utility_pilot_20261029"))
    parser.add_argument("--updates", type=int, default=3000)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.final:
        run_final(args.output, args.updates)
    else:
        run_pilot(args.output, args.updates)


if __name__ == "__main__":
    main()
