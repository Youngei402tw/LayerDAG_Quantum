# Actual status

Final execution status: **TARGET-CONDITIONING GATE PASSED; COMPILER REMAINS THE PRACTICAL WINNER**.

Implemented and verified:

- Independent complex128 statevector simulator cross-checked with Qiskit for ordered X/H/SX/RY/RZ/CX/SWAP/CSWAP/CCX semantics, continuous angles and global phase.
- Typed version-1 logical constraint evaluator with explicit unsupported, inconsistent, qualified, exhausted and incomplete outcomes; exact/min/max operation counts, total gates, dependency depth, two-qubit depth, ordered operands/connectivity and zero ancillas.
- Expanded reset-free Qiskit StatePreparation seeds, level-0/level-3 compiler controls, bounded classical deletion/RZ refinement and deterministic gate-set repair.
- Canonical earliest-layer RY/RZ/CX witnesses, phase-invariant target grouping, train/validation duplicate rejection, target/prefix density features, remaining budgets, layer-zero STOP and hard legal-frontier masks.
- Portable native-PyTorch LayerDAG-derived encoder with labelled wire dependencies, per-wire frontier pooling and dense-oracle aggregation tests; factorized masked frontier prediction; 16-bin periodic angles with learned bounded residuals.
- Checkpointed CPU training, immutable per-request sampling streams, K=1/K=32 tiny evaluation and paired correct/permuted/null conditioning diagnostics with target-level bootstrap intervals.

Evidence:

- 29 tests pass. All 240 compiler-pilot rows replay against Qiskit; maximum original fidelity discrepancy is 1.11e-15.
- Corrected tiny learning gate: DAG repaired 31/32 damaged suffixes and grid 31/32 at K=32; no damaged prefix qualified unchanged and every sampled circuit met its budgets. From scratch, DAG reached 32/32 and grid 31/32 at K=32.
- Held-out 32-target condition check after 128-target development training: grid passed (mean correct-minus-permuted +0.1300, bootstrap 95% interval [+0.0133, +0.2407]); width-64 DAG failed (+0.0327, [-0.0704, +0.1304]). The final width-80 DAG diagnosis also failed (+0.1018, [-0.0073, +0.2118]).
- Audit found label leakage in checkpoint scoring and 44% of training examples, a 1,091-way joint action head with little parameter sharing, mean pooling that discarded the active wire frontier, and severe small-data memorization. The first repaired 256-target run and its classifier-free guidance sweep still failed held-out conditioning.
- Scaling the unchanged repaired model to 2,048 fresh training targets passed on 128 validation targets (+0.1061, 95% CI [+0.0663, +0.1456]) and replicated on the untouched 128-target final split (+0.1045, [+0.0616, +0.1490]). Final mean best-of-8 fidelity was 0.7260 correct, 0.6216 permuted and 0.6355 null; 9/128 correct-condition targets reached 0.99.
- A broader representation study compared matched DAG, raw-sequence, canonical-grid and Transformer encoders for depth-critical gate prediction. The frozen three-seed final test used 87,745-parameter DAG and 88,417-parameter grid models with 6,000 updates each. The grid won on same-size circuits (AUROC 0.9406 vs 0.8556), while DAG won on every larger-circuit paired seed: mean AUROC 0.8078 vs 0.7861 (+0.0217) and F1 0.6341 vs 0.6055 (+0.0286).
- A fresh structured-family replication used five paired seeds and balanced QAOA-like, VQE and arithmetic circuits. DAG won simultaneous width-and-depth shift in all five seeds: AUROC 0.8902 versus 0.8495 (+0.0407), F1 0.8351 versus 0.7928. Width shift produced +0.0514 AUROC while depth-only shift produced +0.0025. On held-out chemistry, grid AUROC was higher by 0.0266; family generalization is not established.
- Downstream rewrite ranking had mixed outcomes. Arbitrary deletion ranking was saturated by a smallest-angle heuristic. For known-safe identity rewrites, DAG recovered 51.94% of oracle depth reduction versus 48.15% for grid and 17.93% random, winning all three seeds. Its +3.79-point mean gain missed the frozen +5-point confirmatory threshold, so this remains supporting evidence rather than a passed final claim.
- A routed structured-circuit development benchmark verified natural-template rewrites by statevector and ranked their downstream line-routing benefit. After correcting a clipped recovery metric and adding a shared candidate-aware readout, DAG removed 6.305 routed CX gates versus 6.664 for grid. The development gate failed and no final test was opened. DAG still beat source criticality (3.266), source-depth lookahead (5.383) and random choice (1.813).
- The first learned DAG run is retained under `experiments/tiny_learning_20261021_invalid_noncanonical` and is invalidated because declared layers did not always equal earliest physical dependency layers.

Consequences:

- Held-out target dependence is now established for this repaired DAG model and data distribution. The result is one training seed and does not establish a DAG-over-grid representation advantage.
- A DAG-over-grid advantage is established specifically for size-shifted compiler critical-path prediction across three seeds. It does not extend to in-distribution prediction, where the canonical grid is stronger, or establish better state-preparation fidelity.
- The compiler-utility result motivates natural rewrite experiments but does not yet establish production optimization benefit because candidates were injected identities and the confirmatory effect-size gate failed.
- Classifier-free guidance scales above one did not help. The final 0.99 success rate remains low, and the exact compiler is still far stronger. The 2x2 guide/representation matrix, multi-seed matrix, internal-window learning, 3/4-qubit scaling, hardware/noise scopes, parser, readout and dynamic circuits remain unrun.
- Checkpoints contain model, optimizer and RNG state, but the current CLI does not resume an interrupted training loop from its exact data position; completed checkpoints support evaluation replay. Faithful mid-training resume remains an engineering gap.
- PyTorch 2.8.0+cpu is used; DGL runtime parity with upstream is unverified. Upstream remains clean at `bc22fed6f112c3eef2e991551bb0677038df3467`.

Internally timed compute-job wall time is at least 11,437.00 seconds (190.62 minutes); routed oracle labeling and the guidance sweep are not fully timed and are excluded from that lower bound. Failed setup/replay attempts and engineering time were not fully timed. Historic consumption before this workspace remains unknown, so this ledger does not certify the cumulative six-hour allowance. No GPU, paid resource, publication or background job was used.

The paper draft is under `paper/`. The strongest next research step is replication of the passed size-transfer claim on public circuit suites, at least ten seeds, larger widths, held-out families and multiple hardware graphs. A new practical compiler claim should target a hardware-aware cost that is expensive to compute from the source DAG; the present routed development result is negative.
