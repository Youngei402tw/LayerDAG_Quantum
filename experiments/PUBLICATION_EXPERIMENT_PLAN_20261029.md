# Publication experiment plan: natural compiler rewrites

## Question

Does explicit wire-dependency structure help a learned compiler prioritize useful, semantics-preserving rewrites when circuit size and circuit family change?

The existing result answers a narrower question: DAG message passing transfers better than a canonical layer grid when identifying critical gates in larger random circuits. The injected-RZ(0) experiment suggests downstream utility, but it is too synthetic for a paper's main compiler claim and missed its frozen effect-size gate.

## Benchmark

Use circuits from at least four families with different structure: QAOA, hardware-efficient VQE, reversible arithmetic, and chemistry ansatz circuits. Split by both family and scale. Training uses small instances from three families; tests include larger instances from those families and a held-out fourth family. Preserve the original logical circuit, target unitary or observable, coupling map, and compiler configuration for replay.

Generate candidates using fixed, production-style passes rather than injected identities:

1. commutative cancellation and inverse-pair removal;
2. one- and two-qubit block resynthesis;
3. rotation merging and zero-angle elimination produced by parameter binding;
4. routing substitutions and local SWAP cancellation;
5. template rewrites exposed by Qiskit optimization passes.

Each candidate must be verified by exact unitary/statevector comparison where tractable, or by a separately specified equivalence checker at larger scale. Unsafe or unverified candidates remain in the ranking pool but cannot count as successful rewrites.

## Compared methods

- wire-dependency DAG ranker;
- canonical earliest-layer grid ranker with parameter count and update count matched to the DAG;
- production compiler order/score;
- local hand-engineered features such as gate angle, block size and immediate depth delta;
- random ranking;
- exhaustive oracle on instances where all candidates can be checked.

All learned methods receive the same gate, operand, angle, hardware and candidate features. Validation selects checkpoints and any ranking threshold. Final circuits, families and random seeds remain unopened until the configuration is frozen.

## Primary endpoint

Mean realized two-qubit-depth reduction per expensive equivalence or simulation call at a fixed verification budget. This endpoint tests the practical reason to rank candidates: useful rewrites should be found before the verification budget is exhausted.

Secondary endpoints are total depth reduction, CX/ECR count reduction, probability of finding any safe improving rewrite, regret relative to exhaustive oracle, wall time, compile time, and final fidelity/equivalence. Report results separately for same-size, larger-size and held-out-family tests.

## Statistical design

Run at least five paired training seeds on identical data and evaluation candidates. Treat circuits as the sampling unit. Report paired bootstrap confidence intervals over circuits and seed-level paired differences. Predeclare a smallest practically meaningful effect after a small development set, then freeze it before final data generation. Include family-stratified results so one easy family cannot dominate the average.

The main claim passes only if DAG beats the canonical grid and the strongest nonlearned heuristic on the primary endpoint for every training seed or under a predeclared hierarchical test, while preserving equivalence. Same-size performance may favor the grid; the intended claim is transfer under scale or family shift.

## Ablations needed for mechanism

- remove dependency edges while preserving node features and parameter count;
- replace directed wire edges with an undirected graph;
- vary message-passing depth;
- randomize legal topological order;
- provide the grid with explicit predecessor/successor features;
- evaluate by circuit width and critical-path length.

These tests distinguish a benefit from dependency structure from extra capacity, accidental ordering cues, or easier optimization.

## Current status and stop rule

Infrastructure for paired DAG/grid training, critical-path labels, exact statevector deletion labels, candidate ranking, oracle depth scoring, deterministic data generation and replay is implemented and tested. The controlled downstream experiment is complete. Natural candidate extraction, real-family loaders and equivalence checking are the remaining implementation work.

Do not present the injected-identity result as confirmatory. Do not revise its +0.05 threshold. If the natural-rewrite development study shows no advantage over production heuristics, retain it as a negative result and narrow the paper to representation transfer rather than compiler improvement.
