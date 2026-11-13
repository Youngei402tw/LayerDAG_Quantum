# DRAFT (not frozen, not run): width extrapolation on standard-algorithm circuits

Status: **proposal awaiting project-owner approval.** A prototype data generator (`qdag/algorithm_suite.py`) and data statistics exist (`experiments/qai_algorithm_suite_scoping_20261113/`). **No model has been trained or evaluated on these circuit families.** The protocol becomes binding only when the owner approves it, it is renamed to drop `DRAFT_`, and it is committed and pushed before the run code is written. Post-confirmatory with respect to the paper's primary gate, which stays as recorded.

## Questions

Do the three conclusions of the synthetic study hold on standard algorithm families compiled by a real transpiler?

- Q-R1: name-free wire connectivity (role DAG, pointer GRU) beats a canonical serialization without connectivity (grid GRU) at 13–20 qubits.
- Q-R2: the pointer GRU ties the role DAG at width but is not invariant to relabeling.
- Q-R3: with relabeling augmentation the pointer GRU becomes nearly invariant in AUROC and trails the DAG slightly.

Phase B asks the same of a label that depends on gate semantics.

## Data

Generator `qdag.algorithm_suite.generate_suite` (Qiskit 2.2.3 circuit library; `transpile(basis_gates=[h, ry, rz, cx], optimization_level=1, seed_transpiler=0)`, no coupling map). Eleven families in equal shares: `qft, qpe, bv, graphstate, qaoa, vqe, trotter, adder, qv, iqp, excitation`. Circuits under 8 gates and exact duplicates are rejected.

| Split | Base seed | Circuits | Qubits |
|---|---|---|---|
| training | 15100 | 1,100 | 4–7 |
| validation | 15101 | 264 | 4–7 |
| same | 15102 | 528 | 4–7 |
| near width | 15103 | 528 | 8–12 |
| far width | 15104 | 528 | 13–20 |

Validation and same-size circuits identical to a training circuit are rejected. Evaluation-only variants: same and far width under an independent uniformly random qubit permutation per circuit (generators 15105 and 15106), and under qubit-order reversal (the endianness convention that differs between frameworks). Content hashes of every split are written before training starts. Seeds 15100–15106 are generated for the first time by the frozen run; scoping used 90100–90104 only.

## Labels

- **Phase A (primary):** membership in a longest wire-dependency path, the paper's probe (`critical_labels`).
- **Phase B (secondary):** membership in a longest chain of Qiskit's commutation-aware dependency DAG (`circuit_to_dagdependency` on the compiled circuit; `algorithm_suite.commutation_labels`). Invariant to relabeling because commutation does not depend on qubit names.

## Models and training (no code or hyperparameter changes, zero tuning)

Role DAG (`InvariantDag`, 80,833), grid GRU (`InvariantGrid`, 84,961), pointer GRU (`PointerGrid`, 87,457), augmented pointer GRU (same, trained on freshly relabeled copies as in `QAI_AUGMENTATION_CONTROL_PROTOCOL_20261113.md`). In Phase B, optionally the role-ablated DAG. Training as in the frozen final: AdamW 3e-4, batch 16, clip 1.0, positive-class weight from the training labels, 4,000 updates, validation-AUROC checkpointing every 250 updates, paired seeds 111–115, 2 threads. Each model is run once per seed. The Transformers are excluded because circuits exceed their 192 learned positions. The pointer distance feature keeps its `log1p(192)` normalization and may exceed 1 on long circuits.

## Metrics

- **Primary:** family-macro node AUROC at far width: the mean over families of the node-pooled AUROC within each family. A family whose labels are single-class in a split is left out of that split's macro and named.
- Secondary: node-pooled AUROC (as in the synthetic study), F1 at 0.5, per-family AUROC table, relabeling gaps (same − relabeled, far − far-relabeled, and the reversal counterparts), maximum logit change and mean absolute probability change on the first 64 circuit pairs of each renaming.
- Paired differences over seeds with a descriptive Student-t interval (df = 4). Fixed test sets; intervals cover training randomness only.

## Pre-declared readings (per phase; thresholds fixed at freeze)

- **R1, connectivity:** *replicated* if both the role DAG and the pointer GRU exceed the grid GRU at far width in all 5 seeds with mean difference ≥ +0.03; *not replicated* if either mean difference is < +0.01; otherwise *partial*.
- **R2, DAG versus pointer GRU:** *tie* if the absolute mean difference is < 0.01; *graph advantage* if the DAG wins all 5 seeds with mean ≥ +0.02; *sequence advantage* if the pointer GRU wins all 5 seeds with mean ≥ +0.02; otherwise *intermediate*.
- **R3, augmentation:** *suffices* if both random-relabeling AUROC gaps are ≤ 0.005 and the DAG's far-width lead is ≤ 0.01; *insufficient* if a gap or the lead is ≥ 0.02; otherwise *intermediate*.
- **R4, invariance:** the role DAG's maximum logit change must be ≤ 1e-6 for both renamings on every checked pair; a failure is an implementation error to report, not to tune away.
- Phase B only, if the role-ablated DAG is run: *roles contribute* if the full DAG beats it by ≥ 0.01 in all 5 seeds; *roles not needed* if the mean difference is < 0.01; otherwise *intermediate*.

The paper reports every reading, including any that contradict the synthetic study.

## Optional external set (needs separate owner approval)

MQT Bench algorithm-level circuits at 13–20 qubits, generated in an isolated environment, exported as OpenQASM, compiled with the pinned Qiskit as above, and used for evaluation only. Descriptive; no reading attached.

## Compute estimate and stopping rule

Measured on untrained models: about 471, 863, 881 and 899 s per seed for the four models. One phase is about 4.3 CPU-hours sequentially and about 2.2 hours as two parallel 2-thread processes; the role-ablated DAG adds about 40 minutes. Five seeds, run once. No reruns, extra seeds, or changes without a new dated protocol.

## Output

`experiments/qai_algorithm_suite_<date>/` (one result file per phase), registry entries and a ledger update. Never write into frozen directories.
