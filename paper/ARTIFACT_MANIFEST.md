# Publication artifact manifest

## Confirmatory results

- QAI invariant far-width final: `experiments/qai_invariant_final_20261102/final_result.json`
  - five paired seeds 111--115;
  - 1,000 train, 256 validation, four 512-circuit final splits;
  - train 4--7 qubits, test through 13--20 qubits;
  - 4,000 updates per model;
  - exact qubit-relabel invariance and frozen gate passed.
- QAI canonical Transformer control: `experiments/qai_transformer_control_20261106/control_result.json`
  - fixed post-confirmatory descriptive control;
  - five seeds and identical data;
  - not part of the primary gate.

- Random-circuit size shift: `experiments/schedule_dag_final_20261028/final_result.json`
  - three paired seeds 51--53;
  - 1,000 train, 256 validation, 512 same-size test, 512 larger test;
  - 6,000 updates per model;
  - frozen gate passed.
- Structured-family size shift: `experiments/structured_transfer_final_20261102/final_result.json`
  - five paired seeds 81--85;
  - 1,000 train, 256 validation, five 512-circuit final splits;
  - 4,000 updates per model;
  - frozen gate passed.

Dataset content hashes are embedded in both result files. Full run metadata and result-file hashes are in `runs/registry.jsonl`. Compute accounting is in `experiments/budget_ledger.json`.

## Reproduction commands

From the repository root with the recorded Python environment:

```sh
PYTHONPATH=. python -m pytest -q
PYTHONPATH=. python -m qdag.schedule_benchmark \
  --final --updates 6000 --output experiments/schedule_dag_final_reproduction
PYTHONPATH=. python -m qdag.structured_transfer \
  --final --output experiments/structured_transfer_final_reproduction
PYTHONPATH=. python -m qdag.invariant_benchmark \
  --final --output experiments/qai_invariant_final_reproduction
PYTHONPATH=. python -m qdag.invariant_benchmark \
  --transformer-control --output experiments/qai_transformer_control_reproduction
```

The generators and training loops are deterministic for the recorded software stack. CPU kernels on a different PyTorch build may produce small floating-point differences.

## Manuscripts

- `qce_paper.pdf`: IEEE QCE-oriented compiler/software framing.
- `qai_paper.pdf`: active IEEE QAI symmetry-aware OOD-learning submission.
- `qai_submission_v1.pdf`: byte-identical versioned QAI review copy.
- `main.pdf`: shared neutral manuscript.

All three PDFs are built from portable two-column LaTeX. Transfer them into the current official venue template only after confirming the live call's page limit, anonymity policy, and artifact-link rules.

## Environment

- Python 3.12
- PyTorch 2.8.0+cpu
- NumPy 2.3.4
- Qiskit 2.2.3
- pytest 8.4.2
- GPU usage: none

The implementation uses native PyTorch sparse `index_add_` aggregation. It is source-informed by LayerDAG but does not claim runtime parity with the upstream DGL implementation.
