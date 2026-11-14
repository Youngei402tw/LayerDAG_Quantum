# Fixed post-confirmatory protocol: width extrapolation on standard-algorithm circuits

Status: **frozen 2026-11-14, approved by the project owner ("Go": both phases, role-ablated DAG in Phase B, no MQT Bench).** Committed and pushed **before the run code is written and before any model is trained or evaluated on these circuit families.** The data generator (`qdag/algorithm_suite.py`) and data-only statistics on throw-away seeds exist (`experiments/qai_algorithm_suite_scoping_20261113/`); the only model code that has touched these families timed untrained networks and computed no accuracy. Post-confirmatory with respect to the paper's primary gate, which stays as recorded. Supersedes `DRAFT_QAI_ALGORITHM_SUITE_PROTOCOL_20261113.md` (renamed into this file; see git history).

## Changes from the 2026-11-13 draft (made before any model was trained)

1. **`bv` family replaced by `oracle`.** A data-only check found only 64 usable distinct Bernstein–Vazirani circuits at 4–7 qubits, but training, validation and same-size need 172 unique ones. The family now draws Bernstein–Vazirani or, with probability one half, Deutsch–Jozsa with a balanced oracle wrapped in a random input mask.
2. **Uniqueness is strict.** The generator raises instead of ever accepting a duplicate. A full-size dry run on throw-away seeds 90200–90204 succeeded (`full_size_check.json`).
3. Relabeling pair statistics use all 528 circuit pairs per renamed split, not the first 64.
4. Every reading below is evaluated on the primary metric; the draft left this implicit.
5. Up to four training processes may run in parallel, each with 2 threads.

## Questions

Do the three conclusions of the synthetic study hold on standard algorithm families compiled by a real transpiler?

- Q-R1: name-free wire connectivity (role DAG, pointer GRU) beats a canonical serialization without connectivity (grid GRU) at 13–20 qubits.
- Q-R2: the pointer GRU ties the role DAG at width but is not invariant to relabeling.
- Q-R3: with relabeling augmentation the pointer GRU becomes nearly invariant in AUROC and trails the DAG slightly.

Phase B asks the same of a label that depends on gate semantics, and whether operand roles matter there.

## Data

Generator `qdag.algorithm_suite.generate_suite` (Qiskit 2.2.3 circuit library; `transpile(basis_gates=[h, ry, rz, cx], optimization_level=1, seed_transpiler=0)`, no coupling map). Eleven families in equal shares: `qft, qpe, oracle, graphstate, qaoa, vqe, trotter, adder, qv, iqp, excitation`. Circuits under 8 gates are rejected.

| Split | Base seed | Circuits | Qubits |
|---|---|---|---|
| training | 15100 | 1,100 | 4–7 |
| validation | 15101 | 264 | 4–7 |
| same | 15102 | 528 | 4–7 |
| near width | 15103 | 528 | 8–12 |
| far width | 15104 | 528 | 13–20 |

Splits are generated in this order and no circuit may equal any circuit generated earlier (in the same or a previous split). Evaluation-only variants of the same and far-width splits: an independent uniformly random qubit permutation per circuit (`np.random.default_rng(15105)` for same, `15106` for far), and qubit-order reversal (q → n−1−q, the endianness convention that differs between frameworks). Renamed circuits carry the labels of their originals, which is exact for both labels because neither depends on qubit names. A manifest with content hashes and per-family statistics of every split is committed before training starts. Seeds 15100–15106 are generated for the first time by the frozen run.

## Labels

- **Phase A (primary):** membership in a longest wire-dependency path, the paper's probe (`critical_labels`).
- **Phase B (secondary):** membership in a longest chain of Qiskit's commutation-aware dependency DAG (`circuit_to_dagdependency` on the compiled circuit; `algorithm_suite.commutation_labels`).

## Models and training (no model code or hyperparameter changes, zero tuning)

Role DAG (`InvariantDag`, 80,833 parameters), grid GRU (`InvariantGrid`, 84,961), pointer GRU (`PointerGrid`, 87,457), augmented pointer GRU (same class; every sampled training circuit replaced by a copy under a fresh uniformly random qubit permutation from `np.random.default_rng(seed + 10000)`, as in `QAI_AUGMENTATION_CONTROL_PROTOCOL_20261113.md`). Phase B additionally trains the role-ablated DAG (`RoleAblatedDag`, 80,833). Training as in the frozen final: AdamW 3e-4, batch 16, clip 1.0, positive-class weight from the phase's training labels, 4,000 updates, checkpoint by node-pooled validation AUROC every 250 updates, paired seeds 111–115 with the frozen minibatch-index stream, 2 threads. Each model is run once per seed. All models are trained in one software environment (Python 3.13.18, PyTorch 2.8.0+cpu, NumPy 2.3.4, Qiskit 2.2.3); no recorded values from other studies enter any comparison. The Transformers are excluded because circuits exceed their 192 learned positions. The pointer distance feature keeps its `log1p(192)` normalization and may exceed 1 on long circuits.

## Metrics

- **Primary:** family-macro node AUROC at far width: the mean over families of the node-pooled AUROC within each family. A family whose labels are single-class in a split is left out of that split's macro and named (expected for `oracle` in Phase B, where the dry run found every gate positive).
- Secondary: node-pooled AUROC (as in the synthetic study), F1 and accuracy at 0.5, the per-family AUROC table, relabeling gaps on the primary metric (same − same-relabeled, far − far-relabeled, and the reversal counterparts), and the maximum logit change and mean absolute probability change over all 528 circuit pairs of each renamed split.
- Paired differences over seeds with a descriptive Student-t interval (df = 4). Test sets are fixed; intervals cover training randomness only.

## Pre-declared readings (each phase separately; primary metric)

- **R1, connectivity:** *replicated* if both the role DAG and the pointer GRU exceed the grid GRU at far width in all 5 seeds with mean difference ≥ +0.03; *not replicated* if either mean difference is < +0.01; otherwise *partial*.
- **R2, DAG versus pointer GRU:** *tie* if the absolute mean difference is < 0.01; *graph advantage* if the DAG wins all 5 seeds with mean ≥ +0.02; *sequence advantage* if the pointer GRU wins all 5 seeds with mean ≥ +0.02; otherwise *intermediate*.
- **R3, augmentation** (augmented pointer GRU): *suffices* if the mean random-relabeling gaps at same size and at far width are both ≤ 0.005 and the DAG's mean far-width lead is ≤ 0.01; *insufficient* if either gap or the lead is ≥ 0.02; otherwise *intermediate*.
- **R4, invariance:** the role DAG's maximum logit change must be ≤ 1e-6 on every pair of all four renamed splits in every seed; a failure is an implementation error to report, not to tune away.
- **Roles (Phase B):** *roles contribute* if the role DAG beats the role-ablated DAG by ≥ 0.01 in all 5 seeds; *roles not needed* if the mean difference is < 0.01; otherwise *intermediate*.

The paper reports every reading, including any that contradict the synthetic study.

## Compute estimate and stopping rule

Measured on untrained models (2 threads): about 471, 863, 881 and 899 s per seed for the four Phase A models; Phase B adds the role-ablated DAG. About 9.3 CPU-hours for both phases if run one model at a time. Five seeds, run once. A unit that dies from an infrastructure failure (process killed, machine restarted) is restarted from scratch with the same seed and this is recorded; a unit that completes is never rerun. No extra seeds or changes without a new dated protocol.

## Output

`experiments/qai_algorithm_suite_20261114/` (data manifest, one file per model and seed, one aggregate per phase), registry entries and a ledger update. Never write into frozen directories.
