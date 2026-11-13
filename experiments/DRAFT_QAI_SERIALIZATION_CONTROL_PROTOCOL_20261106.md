# DRAFT (not frozen, not run): index-free wire-pointer serialization control

Status: **proposal awaiting project-owner approval.** No code exists for this model, no run has been made, and no result has been observed. It becomes binding only after the owner approves it, renames it to drop `DRAFT_`, and commits it **in a commit that precedes any result commit**. That ordering gives the repository its own timestamp evidence, which the existing QAI protocols lack because they were committed together with their results.

Written 2026-11-06 during migration, after the primary DAG-versus-grid final and the Transformer control had been opened. Whatever this control finds, it is post-confirmatory. It can neither rescue nor overturn the frozen primary gate.

## Reviewer concern addressed

In the primary study, the grid GRU and the Transformer receive seven index-free gate features in canonical order and **no explicit shared-wire connectivity**. The DAG's advantage could therefore reflect access to connectivity rather than the dependency-graph inductive bias. The earlier structured study, which gave both models absolute wire IDs, argues against a pure information explanation up to 12 qubits. Absolute IDs cannot extend to 13–20 qubits, however.

## Hypothesis

H1: On the frozen far-width split (13–20 qubits), the role DAG's AUROC exceeds that of an index-free sequence model that receives explicit relative wire pointers and operand roles.

## Data

Exactly the frozen QAI final splits, regenerated from base seeds 14100–14105. The six content hashes must match `experiments/qai_invariant_final_20261102/final_result.json` before training starts. No new data.

## Architecture (fixed before any run)

Use the same canonical earliest-layer order and the same two-layer bidirectional GRU (width 96) as `InvariantGrid`. Each token's 7 features are extended with 2 operand slots × 9 features = 18 features, giving 25 per token. Slot 0 is the single-qubit or control operand; slot 1 is the target operand, or zeros.

For each operand wire of token *i*:

- one-hot role at *i* (3);
- predecessor on the same wire: exists flag (1), its role one-hot (3), and `log1p(i - j) / log1p(192)`, where *j* is the predecessor's canonical position (1).

Wire identity is used only to find the predecessor and is never emitted as a feature. All features are invariant to a consistent relabeling of qubits. The token order still changes with relabeling (wire tie-break), so the model is not exactly invariant. Report this.

Only `node_in` changes size. The expected parameter count is about 86.3k, to be confirmed and recorded before training.

## Training and tuning budget

Identical to the frozen final: AdamW 3e-4, batch 16, clip 1.0, pos-weight, 4,000 updates, validation AUROC checkpointing every 250 updates, paired seeds 111–115. **Zero tuning.** One configuration, run once.

## Metrics

Same, near-width, far-width, and relabeled AUROC/F1 (mean ± s.d. over seeds), and paired far-width AUROC differences against the recorded DAG values for the same seeds.

## Pre-declared reading (thresholds fixed now)

- **Topology supported:** DAG far-width AUROC exceeds the pointer GRU in all 5 seeds **and** the mean difference is ≥ +0.02.
- **Information explains the gap:** mean difference < +0.01, or the pointer GRU wins in at least 3 of 5 seeds. The paper must then say that the DAG advantage over sequences is mainly about exposing connectivity without names.
- **Intermediate:** anything else, reported as such.

The paper must report the result regardless of outcome.

## Compute estimate and stopping rule

About 300 s per model on 2 CPU threads (measured for the grid on 2026-11-06), so about 25–30 min for 5 seeds, plus about 5 min of evaluation. Stop after the 5 seeds. No reruns or additional seeds unless a new dated protocol is written.

## Output

New directory `experiments/qai_serialization_control_<date>/`, a new registry entry, and a ledger update. Never write into the frozen directories.
