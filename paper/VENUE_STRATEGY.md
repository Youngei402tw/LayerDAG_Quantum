# Venue strategy

## Primary target

Target the next available **IEEE International Conference on Quantum Artificial Intelligence (QAI)** call whose scope includes learning for quantum software or circuit optimization. The paper's combination of graph representation learning, quantum circuits and compiler generalization fits that audience. Confirm the live call, page limit and double-blind policy before submission.

## Strong alternative

Submit to **IEEE International Conference on Quantum Computing and Engineering (QCE / IEEE Quantum Week)** in a quantum software, compilation or machine-learning track. For QCE, emphasize compiler representation, exact dependency semantics and reproducibility. The current evidence is best suited to a short/research paper unless the benchmark is expanded to public real-circuit suites.

## Stretch targets

An ICML, NeurIPS or ICLR main-track submission would need substantially broader evidence: public circuit corpora, multiple hardware graphs, at least five to ten seeds, comparison with modern graph and long-sequence architectures, and a practical endpoint that beats the strongest compiler heuristic. Relevant quantum-ML workshops can provide feedback before a main-track attempt, but workshop acceptance should not be described as equivalent to archival conference publication.

## Positioning

Use the title and claim in `main.tex`. Do not claim universal DAG superiority or production compiler improvement. Lead with the controlled observation that representation choice reverses under size shift: the canonical grid wins in-distribution, while the dependency DAG wins on wider and longer circuits in every paired seed. Present the failed deletion and routed-rewrite studies as a useful boundary: better structural transfer does not automatically yield a better optimizer.

## Work that would materially strengthen acceptance odds

1. Replicate critical-path transfer on public circuit families and transpiled circuits, not only random logical circuits.
2. Increase the final comparison to at least ten seeds and report hierarchical circuit/seed uncertainty.
3. Evaluate widths beyond 12 qubits and hold out circuit families.
4. Add graph-transformer and position-free set/sequence controls with matched tuning budgets.
5. Predict an expensive hardware-aware compiler cost that cannot be computed exactly from the source DAG, then demonstrate amortized compiler-call savings.
