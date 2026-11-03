# Execution plan

This run follows CODEX_HANDOFF.txt and its 2026-10-20 constraints/refinement addendum. The handoff was supplied after the initial compiler pilot. The addendum makes compiler-seeded refinement primary; from-scratch generation remains a control.

1. Implement a versioned typed logical constraint contract, ordered operation registry, NumPy pure-state evaluator, Qiskit cross-checks, and explicit outcomes.
2. Run expanded reset-free Qiskit seed and optimized compiler baselines and bounded classical suffix deletion / parameter refinement, with separately verified acceptance.
3. Record requests, target arrays, circuits, seed settings, simulator calls, timings, source attribution and outcomes; replay saved results independently.
4. Run the CPU-profile 2-qubit guided-OFF tiny learning gate for a compiler/witness-seeded suffix editor and a from-scratch control. Use a portable native-PyTorch graph aggregation because DGL is absent. Compare a graph-prefix encoder with a canonical-grid prefix encoder only after physical tests pass.
5. Stop after two diagnosis-driven revisions if the tiny gate fails. Do not launch the large GPU matrix in this CPU-only environment.

## Frozen pilot protocol (before test results)

CPU only. No prior experiment ledger was found before this run; historic consumption outside the workspace remains unknown. The handoff caps aggregate CPU compute jobs at six hours, two-qubit development, and 30 minutes per training job. At least 25% of remaining time is reserved for evaluation, replay, and reporting. The initial compiler pilot was capped at 300 wall seconds and completed in seconds. Its 3-qubit cases are baseline-only and predate recovery of the CPU profile; neural work is limited to two qubits.

## Frozen tiny learning protocol

Freeze before training: 32 unique two-qubit RY/RZ/CX witness targets generated with seed 20261021, 2-6 canonical dependency layers, continuous angles, and exact stored witnesses. Group by phase-invariant density-matrix distance at 1e-10; reject near duplicates. Use maximum bounds equal to each witness. Train teacher-forced valid prefixes with guidance and angle refinement off. Evaluate both from-scratch generation and damaged-seed suffix repair, where one or two terminal layers are removed and cannot pass unchanged. K=1 and K=32 use immutable per-request RNG streams. Success requires at least 90% of the 32 training targets at fidelity >= 0.99 by K=32 with every accepted circuit within budget.

The action space is every legal two-qubit canonical frontier: STOP, either directed CX, and one or two disjoint RY/RZ rotations. Rotation angles use 16 periodic bins plus a learned bounded within-bin residual. A categorical absorbing-mask diffusion with eight reverse steps predicts the next frontier. Both encoders receive the same target/prefix density features, budgets, action semantics, denoiser, and residual-angle head. The DAG encoder uses derived labelled wire dependencies and two bidirectional message-passing blocks; the grid encoder uses canonical layer slots. Width is 64. AdamW starts at 3e-4, batch 64, gradient clip 1.0, and deterministic checkpoint selection by teacher-forced validation loss. Each job is capped at 30 minutes and 20,000 updates, with early stopping after the tiny gate passes or loss stalls. No final-test claim is made from this training-set diagnostic.

## Frozen development conditioning protocol

After the tiny gate, generate 128 additional unique canonical two-qubit witness targets with seed 20261023. Reject phase-invariant duplicates within the training set and across the fixed 32-target validation set generated with seed 20261022. Train one DAG and one grid model from scratch with the same architecture, optimizer and batch size, capped at 10,000 updates or 15 minutes each. Checkpoint by teacher-forced training loss because no validation labels are used during training. Evaluate only the fixed validation targets with K=8 matched request RNG streams under correct, cyclically permuted and null target conditions; score every circuit against the original target. The condition-use claim requires the lower endpoint of a 10,000-draw target-level bootstrap interval for paired correct-minus-permuted best-of-8 fidelity to exceed zero. This is a development diagnostic, not a final test or representation benchmark.

The matched development run showed the width-64 DAG underfit relative to the larger grid model. The final diagnosis-driven revision raises only DAG width to 80, bringing parameter count closer to the grid while holding data, optimizer, update cap, wall cap, validation queries and inference streams fixed. This post-validation capacity diagnosis cannot be used as an unbiased DAG-versus-grid comparison. No further model revision or scaling is permitted in this execution.

## User-authorized repair iteration — 27 October 2026

The subsequent request to find and fix the weak result authorizes a new iteration. Preserve the original frozen conclusions. Audit train/inference parity and target use, repair defects before new results, use fresh target-disjoint splits, predeclare each follow-up under `experiments/`, and keep every training job within 15 minutes. The final scaled protocol is frozen in `experiments/SCALED_REVISION_PROTOCOL_20261027.md`.

The subsequent DAG-advantage exploration uses a separate development/test firewall. Development compares multiple encodings on compiler critical-path reasoning. The final size-transfer claim and thresholds are frozen in `experiments/SCHEDULE_DAG_FINAL_PROTOCOL_20261027.md`; fresh final data and three paired seeds are opened only once.

## Downstream compiler-utility stage — 29 October 2026

The passed size-transfer result is followed by two rewrite-ranking experiments. The first uses full-statevector labels for arbitrary gate deletion and is retained as a negative result because a smallest-angle heuristic saturates the task. The second gives each model one chance to choose among exactly safe identity rewrites and measures realized dependency-depth reduction. Its development pilot passed, but the frozen three-seed test missed the predeclared +0.05 DAG-over-grid oracle-recovery threshold despite favoring DAG in every seed (+0.0379 mean). This is supporting evidence only.

The next publishable test is frozen conceptually in `experiments/PUBLICATION_EXPERIMENT_PLAN_20261029.md`: obtain natural rewrite opportunities from real QAOA, VQE, arithmetic and chemistry circuits; verify semantic preservation; compare matched-parameter DAG and canonical-grid rankers with production compiler heuristics; and report depth/CX reduction per expensive verification call on unseen circuit sizes and families. No threshold will be changed after observing final results.

24 targets: 12 each at 2 and 3 qubits; balanced Haar, product, GHZ-family and short random-circuit witnesses. RNG seed 20261021. Save targets before scoring. Threshold 0.999. Qiskit seed_transpiler=17, approximation_degree=1.0, basis rz/sx/x/cx. Each target gets loose and gate-tightening requests; when the level-0 seed contains x, add an x-forbidden gate-set repair request. The tight quota is one fewer gate than the expanded level-0 seed (minimum zero).

A: optimization level 0. B: optimization level 3. C: starts from A, greedily removes a suffix or a single gate, optimizes remaining rz angles by bounded scipy L-BFGS-B, and verifies complete circuits. Maximum 16 deletion proposals, 20 optimizer iterations/proposal, 1,200 simulator calls and 2 seconds editing per target; precompute deterministic common proposal order. Include explicit x -> sx,sx repair as a classical candidate. C is a classical control, not learned or a DAG diffusion result. All use the identical frozen targets and requests; C has extra editing compute, measured explicitly, so this is not a matched-compute superiority experiment. Reuse identical pools across requests and report that cache reuse.

Measure joint success, fidelity, count/depth, repairs, unchanged/origin, calls and cost. No hyperparameter adjustment after looking at the pilot. No artificial cancelling-pair benchmark. Bootstrap or significance claims are outside this pilot.

## Capability matrix

Supported: zero-input pure-state targets; 1-3-qubit operation registry including CSWAP/CCX; logical allowed/forbidden/required and gate/total/depth/two-qubit-depth quotas; ordered connectivity; zero ancillas; alias normalization; ordered wire DAG serialization; independent final verification.

Unsupported explicitly: native/routed/scheduled constraints, noise, ancillas, protected edit windows, angle constraint predicates, terminal readout, dynamic circuits, learned generation, natural-language parser. Registry simulation support does not imply learned support. Implemented boundaries are tracked in STATUS.md.

## Routed rewrite development and paper — 1 November 2026

A structured-circuit routed rewrite benchmark was frozen before development. The first gate was invalidated after discovering that recovery clipped harmful selections. A corrected listwise run and one final candidate-aware revision were retained. The candidate-aware DAG (106,785 parameters) removed 6.3047 routed CX gates on average versus 6.6641 for the grid (100,641 parameters); the development gate failed, so no final routed test was opened. All safe templates passed statevector verification. The conference-style draft in `paper/` centers the earlier frozen critical-path size-transfer result, reports every downstream failure, and targets IEEE QAI or IEEE Quantum Week/QCE after venue-specific formatting and author completion.

## Structured replication and venue manuscripts — 2 November 2026

One frozen development seed on structured QAOA-like, VQE and arithmetic circuits passed the simultaneous width-and-depth transfer gate. A fresh final protocol then used five paired seeds, fresh data, 4,000 updates per model and separate same-size, width, depth, simultaneous, and held-out-chemistry splits. The final gate passed in all five seeds with +0.0407 mean simultaneous-shift AUROC and +0.0423 F1. Width shift (+0.0514 AUROC) explains most of the effect; depth-only shift is +0.0025. Held-out chemistry favors grid by 0.0266 AUROC and remains a negative diagnostic. Separate QCE and QAI manuscripts use identical evidence with compiler and AI-generalization framing respectively.
