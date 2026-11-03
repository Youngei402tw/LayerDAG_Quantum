# qdag-stateprep

This repository contains a verified CPU prototype for constrained two-qubit state preparation, compiler-seeded classical controls, and a small LayerDAG-derived learned frontier generator with a matched grid control.

The repaired scaled model **passes the held-out target-conditioning gate**, while
Qiskit's exact compiler remains the practical winner. On an untouched
128-target final split, correct conditioning reached mean best-of-8 fidelity
0.7260 versus 0.6216 for permuted targets (paired +0.1045, bootstrap 95% CI
[0.0616, 0.1490]). See `REPORT.md` and `STATUS.md`.

A separate frozen three-seed compiler benchmark identifies a specific DAG
advantage: depth-critical gate prediction transfers better to larger circuits.
On 9–12-qubit, 64–96-gate circuits, the wire-dependency DAG reached AUROC
0.8078 versus 0.7861 for a parameter-matched canonical grid, winning all three
paired seeds. The grid remained better in-distribution.

A fresh five-seed replication on structured QAOA-like, VQE and arithmetic
families separated width from depth shift. DAG reached AUROC 0.8902 versus
0.8495 under simultaneous shift and won all five seeds. Width shift produced a
+0.0514 advantage; depth-only shift produced +0.0025. Held-out chemistry
favored grid, so the claim is within-family size transfer rather than universal
out-of-distribution generalization.

In a downstream safe-rewrite test, DAG recovered 51.94% of oracle depth
reduction versus 48.15% for grid and 17.93% for random choice. DAG won all three
seeds, but the +3.79-point mean gain missed the predeclared +5-point
confirmatory threshold. This is supporting evidence; natural rewrite benchmarks
are still required for a paper-level compiler claim.

A structured routed-rewrite development study did not establish a further DAG
advantage: the final candidate-aware DAG removed 6.305 routed CX gates on
average versus 6.664 for grid. Both learned methods beat exact source
criticality, but the routed final test was not opened after the development
gate failed. The conference-style draft and venue strategy are under `paper/`.

Install and verify from `/workspace`:

```sh
bash /workspace/qdag-environment/install.sh
cd /workspace/LayerDAG_Quantum
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 QISKIT_PARALLEL=FALSE /workspace/.venvs/qdag/bin/python -m pytest -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 QISKIT_PARALLEL=FALSE /workspace/.venvs/qdag/bin/python -m qdag.replay
```

Replay a saved learned checkpoint evaluation without training:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /workspace/.venvs/qdag/bin/python -m qdag.learning \
  --output experiments/tiny_learning_20261021 --representation dag --eval-only
```

The direct verified interface is `qdag.core.evaluate(request, target_array, ordered_operations)`. Unknown request semantics fail closed. `qdag.learning` implements the bounded training and evaluation commands recorded in `runs/registry.jsonl`; rerunning them consumes experiment budget and is unnecessary to inspect saved JSON results.

The official upstream checkout is read-only at `/workspace/.cache/qdag/upstream/LayerDAG`, pinned to `bc22fed6f112c3eef2e991551bb0677038df3467`.
