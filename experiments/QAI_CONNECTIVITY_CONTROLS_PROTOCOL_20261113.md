# Fixed post-confirmatory protocol: index-free connectivity controls

Status: **frozen 2026-11-13, approved by the project owner.** This file is committed and pushed **before any code for these models exists and before any result is observed**, so the repository timestamps it independently of its results (the earlier QAI protocols lack that property). It supersedes `DRAFT_QAI_SERIALIZATION_CONTROL_PROTOCOL_20261106.md` (renamed into this file; see git history for the draft text).

Whatever these controls find, they are post-confirmatory. They can neither rescue nor overturn the frozen primary DAG-versus-grid gate (`QAI_INVARIANT_FINAL_PROTOCOL_20261102.md`), which stays as recorded.

## Reviewer concern addressed

In the primary study the grid GRU and the Transformer receive seven index-free gate features in canonical order and **no explicit shared-wire connectivity**. The DAG's advantage could therefore reflect access to connectivity rather than the dependency-graph inductive bias. A second question is whether the DAG's gain needs the operand-role edge attributes or only the graph topology.

## Changes from the 2026-11-06 draft (made at review, before any code or result)

1. The draft gave each operand only a *predecessor* pointer. The DAG passes messages along both forward and backward edges, so each operand now also gets a *successor* pointer. Otherwise the control would see half the connectivity.
2. Added a pointer-Transformer (same pointer features, the frozen Transformer configuration) because a GRU cannot use a distance feature to look up the pointed-to token, whereas attention with learned positions can in principle.
3. Added a role-ablated DAG (edge attributes set to zero) to separate topology from operand roles.

## Hypothesis (confirmatory within this control)

H1: On the frozen far-width split (13–20 qubits), the role DAG's AUROC exceeds that of the pointer GRU.

The pointer-Transformer and role-ablated DAG are **descriptive** within this control; their readings are stated below but carry no gate.

## Data

Exactly the frozen QAI final splits, regenerated from base seeds 14100–14105. The six content hashes must match `experiments/qai_invariant_final_20261102/final_result.json` before training starts; abort otherwise. No new data.

## Pointer features (both pointer models)

Canonical earliest-layer order exactly as in `InvariantGrid`. Each token keeps its 7 gate features and gains 2 operand slots × 13 features = 26, giving **33 per token**. Slot 0 is the single-qubit operand or the CX control; slot 1 is the CX target, or all zeros for single-qubit gates. For each operand wire of token *i* at canonical position *p(i)*:

- one-hot role at *i* (3);
- predecessor on the same wire: exists flag (1), its role one-hot (3), `log1p(p(i) − p(j)) / log1p(192)` (1);
- successor on the same wire: exists flag (1), its role one-hot (3), `log1p(p(k) − p(i)) / log1p(192)` (1).

Wire identity is used only to find the neighbour and is never emitted as a feature, so every feature is invariant to a consistent qubit relabeling. The token order still changes under relabeling (wire tie-break), so the pointer models are not exactly invariant; report this.

## Architectures (fixed now; only the input layer changes)

- **Pointer GRU:** `InvariantGrid` with `node_in` = Linear(33, 96). Expected 87,457 parameters.
- **Pointer Transformer:** `InvariantTransformer` with `node_in` = Linear(33, 64). Expected 89,729 parameters.
- **Role-ablated DAG:** `InvariantDag` with every edge attribute replaced by zeros. Same 80,833 parameter tensors (the role-input weights receive zero input). Exactly relabeling-invariant.

Record actual counts before training; any mismatch with these expectations means an implementation error to fix before training, not a design change.

## Training and tuning budget

Identical to the frozen final: AdamW 3e-4, batch 16, clip 1.0, positive-class weight, 4,000 updates, validation-AUROC checkpointing every 250 updates, paired seeds 111–115, 2 CPU threads, the existing `train()` minibatch stream. **Zero tuning.** One configuration per model, run once.

## Comparison values and environment check

Paired differences use the **recorded** DAG and grid per-seed values from `qai_invariant_final_20261102`. Because this session runs Python 3.13 rather than the recorded 3.12, one recorded seed (112, DAG + grid) is re-trained first as an environment check and its AUROC differences from the record are reported. If any split differs by more than 1e-3 AUROC, the role DAG is also re-trained for all five seeds in this environment and both comparisons are reported, with the same-environment one stated as the primary reading.

## Metrics

Same, near-width, far-width and relabeled AUROC/F1/accuracy (mean ± sample s.d. over seeds); paired far-width AUROC differences against the DAG for the same seeds; maximum relabeling logit change on the first 64 same/relabeled pairs for every model.

## Pre-declared reading (thresholds fixed now)

Pointer GRU (H1):

- **Topology supported:** DAG far-width AUROC exceeds the pointer GRU in all 5 seeds **and** the mean difference is ≥ +0.02.
- **Information explains the gap:** mean difference < +0.01, or the pointer GRU wins in at least 3 of 5 seeds. The paper must then say that the DAG advantage over sequences is mainly about exposing connectivity without names.
- **Intermediate:** anything else, reported as such.

Pointer Transformer (descriptive): same three bands, labeled descriptive.

Role-ablated DAG (descriptive): if its mean far-width AUROC is within 0.01 of the role DAG, say operand roles are not needed for this probe; if it is ≥ 0.01 lower in all 5 seeds, say roles contribute; otherwise intermediate.

The paper must report every result regardless of outcome.

## Compute estimate and stopping rule

About 300 s per pointer-GRU model, about 70 s per pointer-Transformer and about 150 s per DAG on 2 threads: roughly 45 min in total plus the environment check. Stop after the 5 seeds. No reruns, extra seeds or configuration changes unless a new dated protocol is written first.

## Output

New directory `experiments/qai_connectivity_controls_20261113/`, a new `runs/registry.jsonl` entry and a ledger update. Never write into frozen directories.
