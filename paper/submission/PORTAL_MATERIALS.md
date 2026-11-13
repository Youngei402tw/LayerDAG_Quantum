# Submission-portal materials (drafts)

## Title
Qubit-Label-Invariant Dependency Graphs for Within-Family Width Extrapolation in Quantum-Circuit Learning

## Abstract (plain text, identical in substance to the PDF)
Learned quantum-circuit models are usually trained on circuits that are smaller than the circuits they are meant to handle. Grid and sequence encodings expose a gate's layer, token index, or canonical position. These positions are tied to the training size and to the arbitrary names of logical qubits. We encode a circuit as a wire-dependency directed acyclic graph (DAG) whose nodes carry only gate type, angle, and arity. Each edge records the operand roles (single-qubit, control, or target) that the two operations play on their shared wire. No feature names a qubit, so the model's per-gate predictions are exactly invariant to any permutation of logical-qubit labels. We test this inductive bias on an exactly labeled probe task: predicting which gates lie on a maximum dependency path. Models were trained on 4-7-qubit QAOA-like, hardware-efficient VQE, and reversible-arithmetic circuits, and the evaluation criteria were frozen before the final runs. On 13-20-qubit circuits from the same families, the 80,833-parameter DAG reaches AUROC 0.9252 and the 84,961-parameter canonical-grid GRU reaches 0.8807. The DAG wins all five paired seeds (mean +0.0444) and its logits do not change at all under qubit relabeling. A canonical Transformer added after the primary result reaches 0.7471. An earlier study isolates width as the main source of the gain. A held-out chemistry-inspired family and three downstream compiler tasks did not favor the DAG. The evidence supports a narrow claim: the right symmetry improves within-family width extrapolation of dependency reasoning. It does not show unseen-family robustness, a compiler advantage, or general graph superiority.

## Keywords
quantum circuits; graph neural networks; permutation invariance; out-of-distribution generalization; size generalization; critical path

Suggested topic from the QAI call: Quantum Machine Learning (AI for quantum software).

## Contribution statement (≈60 words)
We give an index-free, operand-role wire-dependency graph whose per-gate predictions are provably invariant to logical-qubit relabeling. In a five-seed experiment with success criteria frozen in advance, it improves maximum-dependency-path prediction from 4–7-qubit training circuits to 13–20-qubit test circuits of the same families over near-matched canonical recurrent and attention encoders. Negative results bound the claim.

## Limitations statement (≈90 words)
The probe label is computable exactly in linear time; the study measures representational inductive bias, not practical utility. Circuits are synthetic, noise-free, and limited to four gate types. Transfer to a held-out chemistry-inspired family failed, and three downstream compiler tasks showed no DAG advantage. Five training seeds on fixed test sets bound training variance only. The sequence baselines receive no explicit wire connectivity. A connectivity-preserving, index-free serialization control has been specified but not run. The Transformer control was added after the primary result and was not tuned.

## Cover note (only if the portal asks for one)
This submission studies a symmetry-aware representation for quantum-circuit learning. Its success criteria were fixed before the final runs. We have tried to state the scope precisely: the result concerns within-family width extrapolation on an exactly labeled probe task, and we report the negative results that bound it. Code, dataset hashes, and per-seed results can be provided anonymously if the chairs permit supplementary material.

## Disclosures (placeholders: the owner must complete them)
- Funding: [WITHHELD FOR REVIEW / TO BE SUPPLIED BY AUTHORS]
- Conflicts of interest: [TO BE SUPPLIED BY AUTHORS; CMT asks for domain conflicts]
- Generative AI use (IEEE policy, acknowledgements section): "A large language model (Anthropic Claude) was used to edit and restructure the manuscript text, generate figure-plotting code, and audit the manuscript's numbers against the machine-readable results. The authors take responsibility for all content." [AUTHORS TO CONFIRM OR AMEND; ADD ANY EARLIER AI USE IN THE PROJECT]
- Artifacts: "An anonymized archive containing code, data generators, dataset hashes, frozen protocols, and per-seed results will be made available [URL TO BE SUPPLIED, ANONYMIZED]." Include only if the venue permits links.
- Prior dissemination: [AUTHORS TO STATE ANY PREPRINT; preprints are allowed under QAI 2026 rules]
