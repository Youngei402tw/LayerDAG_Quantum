# Venue strategy

## Active target

Target the next available **IEEE International Conference on Quantum Artificial Intelligence (QAI)** call whose scope includes quantum-circuit learning. The active paper now contributes exact qubit-label symmetry, a frozen five-seed 13--20-qubit OOD test, and matched recurrent/attention controls. Confirm the live call, page limit and double-blind policy before submission; the official site was inaccessible through the execution proxy.

## Archived alternative

The QCE draft is retained but is not under active development. Current work should change only the QAI paper unless the submission strategy changes explicitly.

## Stretch targets

An ICML, NeurIPS or ICLR main-track submission would need substantially broader evidence: public circuit corpora, multiple hardware graphs, at least five to ten seeds, comparison with modern graph and long-sequence architectures, and a practical endpoint that beats the strongest compiler heuristic. Relevant quantum-ML workshops can provide feedback before a main-track attempt, but workshop acceptance should not be described as equivalent to archival conference publication.

## Positioning

Use `qce_paper.tex` for QCE and `qai_paper.tex` for QAI. Do not claim universal DAG superiority or production compiler improvement. Lead with the result shared by both frozen studies: dependency DAG wins within-family size transfer in all eight paired seeds, with the structured decomposition locating most of the gain in circuit-width growth. Present negative chemistry transfer and failed rewrite studies as boundaries: size transfer does not imply family transfer or a better optimizer.

## Work that would materially strengthen acceptance odds

1. Replicate critical-path transfer on public circuit families and transpiled circuits, not only random logical circuits.
2. Increase the final comparison to at least ten seeds and report hierarchical circuit/seed uncertainty.
3. Evaluate widths beyond 12 qubits and hold out circuit families.
4. Add graph-transformer and position-free set/sequence controls with matched tuning budgets.
5. Predict an expensive hardware-aware compiler cost that cannot be computed exactly from the source DAG, then demonstrate amortized compiler-call savings.

## Migration note (2026-11-06)

Official QAI 2026 rules have now been verified; see `submission/COMPLIANCE_REPORT.md`. The full-paper call is closed, the late-breaking call closes 6 Nov 2026 and is non-archival, and no QAI 2027 call exists yet. The phrase "all eight paired seeds" above counts seeds across two separate studies (3 + 5) with different data and architectures. The active manuscript reports them separately and never aggregates them.
