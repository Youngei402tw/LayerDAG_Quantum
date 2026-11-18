# Response to the simulated review of v5 (manuscript v6)

Changes are in `paper/qai_paper.tex` (v6, built as `paper/qai_submission_v6.pdf`, still 6 pages). Every new number is derived from existing result files and is checked by `scripts/verify_claims.py` and `scripts/check_manuscript_numbers.py`. **No new experiment was run.** Two points can only be fully closed with new experiments (M4 tuning, M5 circuit-level bootstrap); they are marked *partially addressed* and stated as limitations.

| # | Comment | Status | What changed |
|---|---|---|---|
| M1 | Semantic-label claim overgeneralized | **Fixed** | Abstract now says "for the semantic label tested". Conclusion adds "whether this extends to other semantic labels is untested". |
| M2 | Phase A tie depends on metric; pooled values and family counts missing | **Fixed** | Phase A now states that the tie is specific to the pre-registered macro metric. On pooled AUROC the DAG leads both pointer models in all 5 seeds (0.9843 against 0.9532 and 0.9679; +0.0312 and +0.0164). Per-family counts vs the pointer GRU: better on 4, within 0.005 on 4, worse on 3. Phase B: better on 7 of 10 families and within 0.005 on 3, never worse; pooled lead +0.0310 (4 of 5 seeds). The abstract carries the Phase B family result. |
| M3 | Why exact invariance matters, and what it costs | **Fixed** | Introduction gives concrete sources of relabeling (layout and routing renumbering, endianness, mixed dataset conventions). The limitations note the DAG is the smallest model and trained in under 40% of the pointer GRU's time under identical conditions (ratio 0.376 in Phase A, 0.356 in Phase B). |
| M4 | Untuned models | **Partially addressed** | The text states that all models share one recipe without tuning, and gives size and training time. A tuning study with equal budgets would need a new pre-registered experiment; the limitations still say a tuned or relative-position sequence model might narrow the gap. |
| M5 | Uncertainty covers seeds only | **Partially addressed** | Added per-family consistency (M2) as evidence across circuits. A circuit-level bootstrap is not possible from the stored results, because the unit files keep metrics, not per-gate predictions; the limitations now say so. Re-running with stored predictions would need a new run. |
| M6 | Relevance of the labels; credit the commutation DAG | **Fixed** | Introduction: gates on a longest dependency path determine circuit depth. Suite paragraph: the commutation-aware chain length is a lower bound on the depth reachable by reordering commuting gates, and the DAG is credited to Iten et al. (ACM TQC 2022). The citation was verified against arXiv 1909.05270. |
| m1 | Abstract too long | **Fixed** | 288 → 245 words. |
| m2 | "Canonical-grid" name unclear | **Fixed** | Explained as the layer-by-wire grid read layer by layer. |
| m3 | Table II "Relabeled" ambiguous | **Fixed** | Row now reads "Relabeled, 4–7 q". |
| m4 | Transformer paragraph misplaced | **Fixed** | Now its own subsection, "Post-confirmatory Transformer control". |
| m5 | Width and gate count co-vary | **Fixed** | Stated in the data paragraph, with a pointer to the earlier study that separates them. |
| m6 | No availability statement | **Fixed** | Provenance paragraph: code, data, protocols and result files will be released with the final version. *Owner action:* keep that promise, and build an anonymous artifact if the venue allows links (`scripts/make_anonymous_artifact.sh`). |
| m7 | "Reading" undefined | **Fixed** | Evidence-tiers paragraph defines a reading as a named outcome per question, fixed in advance by thresholds on the paired differences. |
| m8 | "Margin grows with width" unquantified | **Fixed** | +0.0073 at 4–7 qubits, +0.0166 at 8–12, +0.0363 at 13–20. |
| m9 | Grid GRU instability not mentioned | **Fixed** | Phase A notes the grid's far-width s.d. 0.0231 against 0.0010 for the DAG. |

## Cuts made to stay at 6 pages

- Figure 1 caption, opening paragraph, relabeling subsection and conclusion: tightened without removing any claim.
- The connectivity and augmentation paragraphs no longer restate every reading threshold, now that readings are defined once. The thresholds stay in the protocol files and in `verify_claims.py`.
- The downstream-tasks paragraph and the acknowledgment were shortened. The AI-use disclosure keeps all four uses.

## Not changed

- Results, readings and tables. No number moved except the new ones listed above.
- The title.
