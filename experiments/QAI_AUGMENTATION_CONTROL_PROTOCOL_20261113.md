# Fixed post-confirmatory protocol: relabeling-augmentation control

Status: **frozen 2026-11-13 under the project owner's standing approval ("continue"), committed and pushed before any code for this control exists and before any result is observed.** Post-confirmatory: it cannot change the frozen primary gate or the connectivity-control readings.

## Question

The connectivity controls (`QAI_CONNECTIVITY_CONTROLS_PROTOCOL_20261113.md`) showed that a pointer GRU with name-free wire pointers ties the role DAG at 13–20 qubits but loses 0.0615 AUROC under qubit relabeling, while the DAG is exactly invariant. Can relabeling augmentation give a sequence model that invariance approximately, without losing width extrapolation? If yes, the DAG's invariance is a convenience; if no, it is a substantive advantage.

## Models (no architecture change, no tuning)

- **Augmented pointer GRU** (primary within this control): `PointerGrid`, 87,457 parameters.
- **Augmented grid GRU** (descriptive): `InvariantGrid`, 84,961 parameters.

## Augmentation

Training is identical to the frozen final (AdamW 3e-4, batch 16, clip 1.0, positive-class weight from the unaugmented training set, 4,000 updates, validation-AUROC checkpointing every 250 updates on the **unaugmented** validation set, seeds 111–115, 2 threads), with one change: every sampled training circuit is replaced by a copy under a fresh uniformly random permutation of its logical qubits (`invariant_benchmark.relabel`). The permutations come from a separate generator `np.random.default_rng(seed + 10000)`, so the minibatch indices equal those of the frozen runs. Labels are unchanged by relabeling (Proposition 1).

## Data

Frozen final splits (base seeds 14100–14105); hashes must match `qai_invariant_final_20261102/final_result.json`. One additional evaluation-only split: **far-width relabeled**, the far-width circuits each under an independent random permutation from `np.random.default_rng(14106)`. It tests whether invariance learned at 4–7 qubits holds at 13–20 qubits. The role DAG is exactly invariant (Proposition 1), so its far-width-relabeled AUROC equals its recorded far-width AUROC; this is stated, not re-trained.

## Metrics

Same, near-width, far-width, relabeled and far-width-relabeled AUROC/F1/accuracy (mean ± s.d.); relabeling gaps `same − relabeled` and `far − far_relabeled`; mean absolute probability change and maximum logit change between each circuit and its relabeled copy on the first 64 same-size and first 64 far-width pairs; paired far-width AUROC differences against the recorded role DAG.

## Pre-declared reading (augmented pointer GRU)

- **Augmentation suffices:** mean same-size relabeling gap ≤ 0.005 **and** mean far-width relabeling gap ≤ 0.005 **and** mean far-width AUROC ≥ recorded DAG − 0.01. The paper must then say exact invariance is matched approximately by augmentation at no accuracy cost on this probe.
- **Augmentation insufficient:** mean same-size or far-width relabeling gap ≥ 0.02, **or** mean far-width AUROC ≤ recorded DAG − 0.02.
- **Intermediate:** anything else.

The augmented grid GRU is descriptive. All results are reported regardless of outcome.

## Compute and stopping rule

About 350 s per model; the two models may run as parallel processes of 2 threads each. Five seeds each, run once. No reruns, extra seeds or changes without a new dated protocol.

## Output

`experiments/qai_augmentation_control_20261113/`, a registry entry and a ledger update. Never write into frozen directories.
