"""Structured circuit-family size-transfer benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .routed_rewrite import FAMILIES
from .schedule_benchmark import CircuitExample, critical_labels, evaluate, examples_hash, train


def generate_structured(seed: int, count: int, qubits: tuple[int, int], lengths: tuple[int, int],
                        families: tuple[str, ...]) -> list[CircuitExample]:
    rng = np.random.default_rng(seed)
    output = []
    for index in range(count):
        family = families[index % len(families)]
        n_qubits = int(rng.integers(qubits[0], qubits[1] + 1))
        length = int(rng.integers(lengths[0], lengths[1] + 1))
        ops = FAMILIES[family](rng, n_qubits, length)
        output.append(CircuitExample(n_qubits, ops, critical_labels(n_qubits, ops)))
    return output


def make_splits(base_seed: int, final: bool) -> dict[str, list[CircuitExample]]:
    count_train, count_val, count_test = (1000, 256, 512) if final else (600, 128, 128)
    seen = ("qaoa", "vqe", "arithmetic")
    return {
        "training": generate_structured(base_seed, count_train, (4, 7), (32, 64), seen),
        "validation": generate_structured(base_seed + 1, count_val, (4, 7), (32, 64), seen),
        "same_size": generate_structured(base_seed + 2, count_test, (4, 7), (32, 64), seen),
        "width_shift": generate_structured(base_seed + 3, count_test, (8, 12), (32, 64), seen),
        "depth_shift": generate_structured(base_seed + 4, count_test, (4, 7), (72, 104), seen),
        "both_shift": generate_structured(base_seed + 5, count_test, (8, 12), (72, 104), seen),
        "heldout_chemistry": generate_structured(base_seed + 6, count_test, (8, 12), (72, 104), ("chemistry",)),
    }


def aggregate(seed_results: list[dict]) -> dict:
    output = {}
    for kind in ("dag", "canonical_grid"):
        output[kind] = {}
        for split in ("same_size", "width_shift", "depth_shift", "both_shift", "heldout_chemistry"):
            output[kind][split] = {}
            for metric in ("auc", "f1", "accuracy"):
                values = np.asarray([row["models"][kind][split][metric] for row in seed_results])
                output[kind][split][metric + "_mean"] = float(values.mean())
                output[kind][split][metric + "_sample_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
                output[kind][split][metric + "_values"] = values.tolist()
    return output


def run(output: Path, final: bool = False, updates: int | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if updates is None:
        updates = 4000 if final else 2000
    splits = make_splits(12100 if final else 11100, final)
    seeds = (81, 82, 83, 84, 85) if final else (71,)
    seed_results = []
    for seed in seeds:
        row = {"seed": seed, "models": {}}
        for kind in ("dag", "canonical_grid"):
            model, training = train(kind, splits["training"], splits["validation"], seed, updates)
            row["models"][kind] = {"training": training}
            for split in ("same_size", "width_shift", "depth_shift", "both_shift", "heldout_chemistry"):
                row["models"][kind][split] = evaluate(model, splits[split])
        seed_results.append(row)
    summary = aggregate(seed_results)
    differences = {
        split: [row["models"]["dag"][split]["auc"] - row["models"]["canonical_grid"][split]["auc"]
                for row in seed_results]
        for split in ("same_size", "width_shift", "depth_shift", "both_shift", "heldout_chemistry")
    }
    both = np.asarray(differences["both_shift"])
    f1_difference = (summary["dag"]["both_shift"]["f1_mean"]
                     - summary["canonical_grid"]["both_shift"]["f1_mean"])
    width_difference = (summary["dag"]["width_shift"]["auc_mean"]
                        - summary["canonical_grid"]["width_shift"]["auc_mean"])
    result = {
        "protocol": "fresh five-seed structured-family final" if final else "frozen structured-family development",
        "updates_per_model": updates,
        "seeds": list(seeds),
        "dataset_hashes": {name: examples_hash(examples) for name, examples in splits.items()},
        "seed_results": seed_results,
        "aggregate": summary,
        "dag_minus_grid_auc": differences,
        "both_shift_auc_mean_difference": float(both.mean()),
        "both_shift_f1_mean_difference": float(f1_difference),
        "width_shift_auc_mean_difference": float(width_difference),
        "gate_passed": bool(np.all(both > 0) and (not final or (
            both.mean() >= 0.015 and f1_difference > 0 and width_difference > 0))),
        "heldout_family_is_diagnostic": True,
    }
    name = "final_result.json" if final else "development_result.json"
    (output / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"aggregate": summary, "dag_minus_grid_auc": differences, "gate_passed": result["gate_passed"]}, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/structured_transfer_development_20261102"))
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--updates", type=int)
    args = parser.parse_args()
    run(args.output, args.final, args.updates)


if __name__ == "__main__":
    main()
