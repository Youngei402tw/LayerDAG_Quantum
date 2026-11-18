# Simulated peer review of manuscript v5

Written 2026-11-18 at the owner's request, as a reviewer for an IEEE quantum-AI conference would write it, before any revision. Every factual point was checked against the result files. The response and the changes made are in `RESPONSE_TO_REVIEW_v5.md`.

**Paper:** *Name-Free Wire Connectivity for Quantum-Circuit Learning: Exact Relabeling Invariance and Width Extrapolation* (6 pages, anonymous)

## Summary

The paper encodes a quantum circuit as a wire-dependency DAG whose edges carry operand roles and whose features never name a qubit, and proves that per-gate predictions are then exactly invariant to qubit relabeling. On a longest-dependency-path probe, a role DAG trained on 4–7-qubit synthetic circuits beats a canonical-order GRU at 13–20 qubits. Pre-registered controls show that a GRU given name-free wire pointers matches the DAG, so the gain comes from exposing connectivity. Relabeling augmentation gives that GRU near-invariance in AUROC but not per gate. A pre-registered study on eleven Qiskit-library algorithm families repeats this for the probe label. For a commutation-aware label, the role DAG leads every sequence model and its operand roles matter.

## Strengths

- **Unusually careful methodology.** Every post-confirmatory protocol was pushed before its code, with readings fixed in advance, paired seeds and hash-checked data. Negative and inconvenient results are reported prominently.
- **The controls ask the right question.** The pointer GRU separates connectivity from graph structure. The authors report its outcome even though it removes the paper's original headline.
- **Real algorithm circuits.** The second study uses real algorithm constructions and a quantum-specific label. It turns a synthetic probe study into a more credible one.
- **Clear invariance result.** The proposition is short and correct, and the measurements confirm it (logit change exactly zero on every renamed circuit).

## Major comments

**M1. Overgeneralized headline about semantic labels.** The abstract and conclusion say the role-labeled graph extrapolates better "when the label depends on what the gates do". The evidence is one such label on one suite. The limitations section says this, but the abstract does not. Please scope the claim to the commutation-aware label tested.

**M2. The Phase A conclusion depends on the metric, and the alternative is not reported.** Phase A declares a tie between the DAG and the pointer GRU on the family-macro AUROC. Node-pooled AUROC was also computed but is not given for the suite. A reader cannot tell whether the tie is robust or an artefact of weighting small families equally with QFT and phase estimation. Please report the pooled values and paired differences next to the macro values, and say how they differ. Please also report how many families favour each model in both phases. The Phase B claim would be much stronger if it held family by family and not only on average.

**M3. Why should exact invariance matter?** Invariance holds by construction, so its value has to be argued. When are circuits actually relabeled in practice? The reversal experiment (endianness) is a good start, but it is buried in Section V-D. Please motivate this in the introduction. Please also state what invariance costs: the DAG is the smallest model, and it appears to train faster.

**M4. Untuned models on all sides.** No model was tuned, and each baseline has one configuration inherited from earlier development runs. That is fair as a protocol, but it limits what "the DAG leads by 0.036" means. Please state clearly that all models share one recipe and no tuning, and give the parameter counts in the suite table or text. Ideally, add a tuning study with equal budgets.

**M5. Uncertainty covers training seeds only.** The intervals come from 5 seeds on fixed test sets, as the authors note. Some evidence of consistency across circuits would help, for example per-family consistency or a circuit-level bootstrap.

**M6. Relevance of the labels.** Both labels can be computed exactly, so the paper studies inductive bias. A sentence on why longest-path membership matters to compilers would help a quantum audience: it identifies the gates that determine circuit depth. So would a sentence on why commutation-aware chains matter: they bound the depth reachable by commutation-based reordering. The commutation-aware DAG used in Phase B should be credited to its source.

## Minor comments

- m1. The abstract is about 290 words; IEEE conference abstracts are usually 150–250. It reads as a list of numbers.
- m2. "Canonical-grid GRU" is confusing for a model that reads a sequence. Explain the name once.
- m3. Table II's "Relabeled" row should say it is the 4–7-qubit split relabeled.
- m4. The Transformer-control paragraph sits inside the relabeling subsection; give it its own heading.
- m5. In the primary splits, width and gate count grow together. Say so, and point to where the two are separated.
- m6. Add a code-and-data availability statement. An anonymous artifact is fine.
- m7. The pre-declared reading names (*tie*, *suffices*, …) appear without a general definition. Define once what a "reading" is.
- m8. "Its margin grows with width" in Phase B should give the near-width number.
- m9. The grid GRU's seed variance in Phase A (s.d. 0.0231) is ten times the DAG's and is not mentioned.

## Recommendation

**Weak accept, conditional on M1–M3.** The methodology is a model for the area, and the honest reframing is a strength. M1–M3 can be fixed in the text with numbers the authors already have. M4 and M5 can be partly addressed now; the rest belongs in a limitations statement. M6 needs a sentence and a citation.

**Confidence:** 4 of 5.
