# Claim-to-evidence table and limitations and negative-results audit

Every numeric value is re-derived by `scripts/verify_claims.py` (handoff values) and `scripts/check_manuscript_numbers.py` (all 85 decimals in the manuscript). Both passed on 2026-11-06.

## Claims made in `paper/qai_paper.tex`

| # | Claim (manuscript wording, condensed) | Tier | Evidence | Status |
|---|---|---|---|---|
| 1 | Role-DAG predictions are exactly invariant to logical-qubit permutations | Theory + test | Proposition 1 (proof in paper); `qdag/invariant_benchmark.py::role_edges`, `invariant_node_features`; `tests/test_invariant_benchmark.py::test_role_features_ignore_absolute_qubit_labels` | Verified |
| 2 | Max relabel logit diff = 0.0 on 320 circuit–relabel pairs; DAG same/relabeled metrics identical | Primary | `final_result.json`: `dag_max_qubit_relabel_logit_difference`, `dag_max_same_vs_relabel_metric_difference` (code checks `same[:64]` per seed) | Verified |
| 3 | Seed-111 re-run is bitwise identical; 0.0 logit diff on all 512 | Integrity (post hoc) | `experiments/qai_invariant_spot_reproduction_seed111_20261106/spot_reproduction.json`; registry entry | Verified 2026-11-06 |
| 4 | Far-width AUROC 0.9252 ± 0.0020 (DAG) vs 0.8807 ± 0.0045 (grid) | Primary | `aggregate.{dag,grid}.far_width.auc_*` | Verified |
| 5 | DAG wins all 5 seeds; mean +0.0444, s.d. 0.0035, t-interval [+0.0400, +0.0488] | Primary | `far_width_auc_paired_differences`; t computed with df = 4 | Verified. The interval reflects training randomness only |
| 6 | Far F1 0.8458 vs 0.8011 (+0.0446) | Primary | `aggregate.*.far_width.f1_mean`, `far_width_f1_mean_difference` | Verified |
| 7 | Grid AUROC drops 0.1097 under relabeling | Primary diagnostic | 0.9391 − 0.8294 | Verified |
| 8 | All five frozen gate conditions pass | Primary | `QAI_INVARIANT_FINAL_PROTOCOL_20261102.md`; `positive_claim_passed: true`; recomputed in `verify_claims.py` | Verified |
| 9 | Transformer 0.8888 / 0.7467 / 0.7471 / 0.7806 AUROC; post-confirmatory | Control | `control_result.json` (`status: descriptive_control_not_part_of_confirmatory_gate`); `QAI_TRANSFORMER_CONTROL_PROTOCOL_20261106.md` | Verified; labeled as post-confirmatory throughout |
| 10 | Parameters 80,833 / 84,961 / 88,065 ("near-matched") | All | `training.parameters` in every seed | Verified |
| 11 | Datasets regenerate to the recorded hashes; no train/test duplicates | Integrity | `scripts/check_dataset_hashes.py` | Verified |
| 12 | Width-only +0.0514, depth-only +0.0025, both +0.0407; chemistry −0.0266 | Supporting (different architecture) | `structured_transfer_final_20261102/final_result.json` | Verified; not pooled |
| 13 | Random-circuit reversal: same 0.8556 vs 0.9406; larger 0.8078 vs 0.7861; seeds +0.0177 to +0.0274 | Supporting | `schedule_dag_final_20261028/final_result.json` | Verified; not pooled |
| 14 | Rewrite gate failed (+0.0379 vs +0.05); routed dev 6.30 vs 6.66 CX; state-prep 9/128 at fidelity 0.99 | Negative | `critical_rewrite_utility_final_20261029/final_result.json`; `routed_rewrite_development_revision2_20261101/development_result.json`; `REPORT.md` table | Verified |

## Claims deliberately NOT made

The paper does not claim, and its discussion and conclusion explicitly rule out:

- universal DAG superiority;
- unseen-family generalization;
- compiler or routing improvement;
- competitive state preparation;
- quantum advantage or hardware execution;
- that the network should replace exact critical-path computation;
- that Q-Fusion's reported results are invalid;
- exact parameter matching;
- population-level significance from 5 seeds.

## Limitations and negative-results audit

| Item | Where in paper | Note |
|---|---|---|
| Probe task is exactly solvable in linear time | §III-D, §VI | Representation probe only |
| Synthetic generated families; 4 gate types; no noise or hardware | §IV, §VI | |
| Held-out chemistry family favors grid; both near chance | Table III, §VI | Rules out family transfer |
| Small-angle deletion saturated by heuristic | §VI | |
| Safe-rewrite gate failed (+0.0379 < +0.05) | §VI | Threshold not changed |
| Routed-rewrite development favored grid; no final opened | §VI | |
| State-prep generator not competitive (9/128 ≥ 0.99) | §VI | |
| **Sequence baselines lack explicit connectivity** | §III-C, Table I, §V-D, §VI | Newly disclosed in this migration. Draft control protocol: `experiments/DRAFT_QAI_SERIALIZATION_CONTROL_PROTOCOL_20261106.md` |
| No relabeling augmentation for baselines | §V-B | Newly disclosed |
| t-interval covers training seeds only; test sets fixed | §IV, §VI | Newly disclosed |
| Transformer untuned, post-confirmatory | §III-C, §IV, §V-C | |
| Protocols share a git commit with their results | §VI (Provenance) | Newly disclosed. The previous draft's "Every protocol predates its corresponding final result" was unverifiable from the repository |
| Positive-label rate 0.52–0.70 | §III-D | Newly reported |
| Compute is a lower bound (14,244.91 s wall clock), not CPU core-seconds | §VI | |

## Discrepancies found in pre-migration records (preserved, not edited)

- `paper/archive/qai_paper_handoff_304144c.tex` says "Every protocol predates its corresponding final result". Git cannot verify this (same commit 304144c).
- `paper/VENUE_STRATEGY.md` counts "all eight paired seeds" across two different studies (3 + 5). That is a count across non-poolable experiments. The new manuscript reports the studies separately.
- `references.bib` (handoff): `li2019tackling` had a wrong DOI (`10.1145/3293882.3336993`) and was typed `@article`. `cowtan2019qubit` omitted author A. Krajenbrink. TFQ was dated 2021 (arXiv v1 is 2020). All are corrected, with the original kept in `paper/archive/references_handoff_304144c.bib`.
- The handoff README points to `/workspace/qdag-environment/install.sh`, which is not in the repository. It is replaced for this paper by `requirements-qai.txt` and `REPRODUCIBILITY.md`.
- No plotting scripts existed for the handoff figures. New figures come from `scripts/make_figures.py`, and the old figure files are retained unchanged.
- Model checkpoints (`*.pt`) are git-ignored and not in the repository. Re-evaluation requires retraining, which is deterministic on the recorded stack (seed 111 verified).
