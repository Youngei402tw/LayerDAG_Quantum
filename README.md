# qdag-stateprep

This repository contains a verified CPU prototype for constrained two-qubit state preparation, compiler-seeded classical controls, and a small LayerDAG-derived learned frontier generator with a matched grid control.

The final experimental status is **SCIENTIFIC GATE FAILED**: physical checks and the tiny repair gate passed, but the DAG model did not demonstrate reliable held-out target dependence after bounded diagnosis. See `REPORT.md` and `STATUS.md`.

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
