# Claim-to-evidence table and limitations and negative-results audit

Every numeric value is re-derived by `scripts/verify_claims.py` (handoff values) and `scripts/check_manuscript_numbers.py` (every decimal in the manuscript; 195 in v5). Both passed on 2026-11-14 for v5; the 2026-11-06 migration build had 85 decimals.

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
| 15 | Pointer GRU ties the DAG at far width: 0.9274 vs 0.9252; paired −0.0023 (s.d. 0.0049, [−0.0083, +0.0038]); DAG wins 2/5; reading *information explains the gap* | Post-confirmatory (protocol pushed before code) | `qai_connectivity_controls_20261113/controls_result.json`; `QAI_CONNECTIVITY_CONTROLS_PROTOCOL_20261113.md` | Verified 2026-11-13 |
| 16 | Pointer GRU relabel drop 0.0615; DAG far F1 higher (0.8458 vs 0.8081, 5/5, not pre-declared) | Post-confirmatory | same | Verified |
| 17 | Pointer Transformer far 0.6900; role-ablated DAG far 0.9199 (−0.0053, [+0.0019, +0.0086], 5/5; below 0.01 rule) | Post-confirmatory, descriptive | same | Verified |
| 18 | Augmented pointer GRU: relabel gaps 0.0035 / −0.0024; mean per-gate |Δp| 0.053, max logit 10.8; far 0.9149; DAG +0.0103 [+0.0055, +0.0150], 5/5; reading *intermediate*. Augmented grid far 0.8657 | Post-confirmatory (protocol pushed before code) | `qai_augmentation_control_20261113/augmentation_result.json`; `QAI_AUGMENTATION_CONTROL_PROTOCOL_20261113.md` | Verified 2026-11-13 |
| 19 | Suite data: 11 families, 1,100 / 264 / 528×3 circuits, up to 1,191 gates at far width, no repeated circuit, labels disagree on 28% (training) and 36% (far) of gates, family sizes differ 33× at far width; carried labels recomputed and equal on every renamed circuit | Post-confirmatory (protocol pushed before run code; manifest before training) | `qai_algorithm_suite_20261114/data_manifest.json`; `QAI_ALGORITHM_SUITE_PROTOCOL_20261114.md` | Verified 2026-11-14 |
| 20 | Phase A: DAG +0.0838 and pointer GRU +0.0825 over grid (5/5 each, *replicated*); DAG − pointer +0.0013 [−0.0033, +0.0060] (*tie*); pointer far gaps 0.0237 (relabel) / 0.0362 (reversal), grid 0.0656 / 0.0828; augmented gaps 0.0005 / −0.0013 and DAG lead +0.0005 (*suffices*); augmented far mean per-gate probability change 0.072; DAG change exactly 0 on 2,112 renamed circuits per seed; far F1 0.9738 / 0.8112 / 0.9004 | Post-confirmatory | `phaseA_result.json` and 20 unit files | Verified |
| 21 | Phase A family pattern: pointer models better on graph states, IQP, QAOA; DAG better on QFT, phase estimation, quantum volume, hardware-efficient ansatz | Post-confirmatory, descriptive | `phaseA_result.json` (`family_auc_mean`); asserted in `verify_claims.py` | Verified |
| 22 | Phase B: DAG − pointer +0.0363 [+0.0303, +0.0422] 5/5 (*graph advantage*); DAG − augmented +0.0421 [+0.0351, +0.0491] with AUROC gaps at most 0.0012 (*insufficient* by the lead criterion); roles +0.0352 [+0.0323, +0.0381] 5/5 (*roles contribute*); same-size 0.9635 vs 0.9562; pointer GRU +0.0748 over grid (*replicated*); oracle family excluded as single-class | Post-confirmatory | `phaseB_result.json` and 25 unit files | Verified |
| 14 | Rewrite gate failed (+0.0379 vs +0.05); routed dev 6.30 vs 6.66 CX; state-prep 9/128 at fidelity 0.99 | Negative | `critical_rewrite_utility_final_20261029/final_result.json`; `routed_rewrite_development_revision2_20261101/development_result.json`; `REPORT.md` table | Verified |

## Claims deliberately NOT made

The paper does not claim, and its discussion and conclusion explicitly rule out:

- universal DAG superiority (the Phase B advantage is stated as one commutation-aware label on one suite with untuned models);
- that either suite label is practically useful to learn (both are exactly computable);
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
