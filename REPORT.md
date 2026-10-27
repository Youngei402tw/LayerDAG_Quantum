# qdag-stateprep experiment report — 27 October 2026

## Outcome

**TARGET-CONDITIONING GATE PASSED AFTER IMPLEMENTATION REPAIR AND DATA SCALING.** The final DAG model shows reproducible held-out target dependence. Its absolute fidelity and 0.99 success rate remain far below the exact compiler, and no DAG-over-grid representation advantage is claimed.

## Compiler-seeded baseline

The frozen compiler pilot used 24 targets at two and three qubits. Acceptance required ideal fidelity at least 0.999 and the declared logical bounds.

| Method | Loose success | One-gate-tighter success | Mean gates | Mean depth | Mean loose fidelity |
|---|---:|---:|---:|---:|---:|
| Expanded Qiskit level 0 | 24/24 | 0/24 | 27.50 | 20.00 | ~1 |
| Qiskit level 3 | 24/24 | 24/24 | 20.54 | 16.04 | ~1 |
| Level 0 plus classical deletion/RZ refinement | 24/24 | 20/24 | 24.83 | 17.33 | 0.999887 |

Optimized Qiskit won this pilot. A supplemental SX-forbidden repair replaced every SX by an exact RZ/RY/RZ decomposition up to global phase and passed 24/24 cases for both compiler levels. It increased gate count and is not compression or learned editing.

## Learned tiny repair gate

The valid tiny set contains 32 unique two-qubit RY/RZ/CX witness targets with canonical earliest dependency layers and continuous angles. The learned action predicts an entire legal next frontier using eight-step absorbing-mask categorical denoising; angles use 16 periodic bins plus a learned within-bin residual. Guidance and angle optimization were off. Damaged repair prefixes were independently verified below 0.99 before generation.

| Model | Params | Updates | Train seconds | Repair K=1 | Repair K=32 | Scratch K=1 | Scratch K=32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DAG width 64 | 184,213 | 5,000 | 209.03 | 31/32 | 31/32 | 25/32 | 32/32 |
| Grid width 64 | 265,301 | 3,100 | 37.98 | 31/32 | 31/32 | 19/32 | 31/32 |

Every sampled circuit respected gate, depth and CX maxima. The DAG's better from-scratch count is one seed on training targets and is not a representation claim. Grid training was substantially faster; equal width did not produce equal parameter count or compute.

The first DAG run is invalid. Its generator serialized some independent operations in consecutive declared layers even though they shared no dependency, so declared layer count and physical depth could diverge. The run is preserved under `experiments/tiny_learning_20261021_invalid_noncanonical`. The corrected generator canonicalizes ordered operations, asserts layer count equals recomputed depth, and rejects phase-equivalent duplicates.

## Target-conditioning evidence

An initial held-out check trained only on the 32-target tiny set gave a positive interval for DAG and an interval crossing zero for grid. Following the handoff, the actual development stage used 128 new training targets and a disjoint fixed 32-target validation set. Each condition used K=8 matched RNG streams and was scored against the original target.

| Model | Params | Updates | Teacher loss | Correct | Permuted | Null | Correct - permuted, 95% CI | Gate |
|---|---:|---:|---:|---:|---:|---:|---|---|
| DAG width 64 | 184,213 | 10,000 | 0.02728 | 0.6284 | 0.5957 | 0.6187 | +0.0327 [-0.0704, +0.1304] | Failed |
| Grid width 64 | 265,301 | 7,800 | 0.01326 | 0.6882 | 0.5582 | 0.5907 | +0.1300 [+0.0133, +0.2407] | Passed |
| DAG width 80 diagnostic | 241,701 | 10,000 | 0.01590 | 0.6376 | 0.5358 | 0.5990 | +0.1018 [-0.0073, +0.2118] | Failed |

The width-80 run was chosen after inspecting validation behavior to address capacity mismatch. It is diagnostic and cannot be used as an unbiased representation comparison. It improved loss and the paired mean, but the interval still crossed zero. These original runs are retained as negative evidence. A later user-authorized iteration audited and repaired the learning path.

## Learning-path audit and repair

The audit found four interacting causes of the weak DAG result:

1. checkpoint scoring supplied the clean action through the noisy-action input, making the reported low loss partly label copying;
2. 44% of training examples also exposed the clean action although generation always began from the absorbing mask;
3. the 1,091-class joint output could not share evidence across the two wires' gate and angle choices, and global mean graph pooling blurred each wire's active frontier;
4. after those fixes, 256 continuously parameterized training targets were still small enough for a 117,848-parameter model to memorize.

The repaired model always predicts from a mask, factorizes action type and per-wire gate/angle logits, pools the last DAG node on each wire, applies 10% null-target dropout, and gives empty-prefix decisions half of each minibatch. A classifier-free guidance sweep over scales 1, 2, 4 and 8 did not help; scale 1 remained best.

The decisive scaled run used 2,048 fresh training targets, 128 validation targets and 128 untouched final targets. All splits reject phase-invariant duplicates. The architecture and optimizer were frozen before this run.

| Split | Correct | Permuted | Null | Correct - permuted, 95% CI | F>=0.99 correct |
|---|---:|---:|---:|---|---:|
| Validation (128) | 0.7398 | 0.6338 | 0.6299 | +0.1061 [+0.0663, +0.1456] | 6/128 |
| Final (128) | 0.7260 | 0.6216 | 0.6355 | +0.1045 [+0.0616, +0.1490] | 9/128 |

These are matched best-of-8 results. They establish target use on this distribution, including replication on the gated final split. They do not show competitive state preparation: only 7.0% of final targets reached fidelity 0.99, versus exact fidelity from the compiler baseline.

## Verification and provenance

- Tests: 21 passed, covering matrices and operand direction, continuous-angle round trips, global-phase density features, layer-zero STOP, hard masks, contradictions/statuses, canonical scheduling, factorized action uniqueness, wire-frontier readout and native index-add aggregation against a dense oracle.
- Compiler replay: 144 original and 96 gate-repair rows verified; 24 target hashes matched; maximum NumPy/Qiskit fidelity difference 1.11e-15.
- Source: official LayerDAG checkout, clean at `bc22fed6f112c3eef2e991551bb0677038df3467`. Upstream code was inspected and kept read-only. The portable model reimplements bidirectional sparse aggregation with native PyTorch; DGL/upstream runtime parity is unverified.
- Runtime: Python 3.12, PyTorch 2.8.0+cpu, Qiskit 2.2.3, NumPy 2.3.4, SciPy 1.16.3, pytest 8.4.2.
- Valid tiny target SHA256: `46436dd2817e3a494c11e4dc7681b2752cbd59899b915a820157a3c94d5532f5`.
- Development record-content hashes: training `5db63fb71b0d9dcb35ac121d9f34582d9bb3c2c03328e40a4c92349933ac69d5`; validation `683d70e7cb81faccd9ce461cfed14e10e7c516269a0f8df716349b9cf99bc64d`.
- Full checkpoint hashes and every valid/invalid job are recorded in `runs/registry.jsonl`.

## Compute and limits

Measured compute-job wall time with internal timers is at least 3,608.07 seconds (60.13 minutes), including the invalidated run, repaired 256-target run, and scaled run. The guidance sweep did not record its own timer and is excluded from that lower bound. Every training job remained below 30 minutes. Failed setup/replay attempts, process startup and engineering time were not all measured, and historic consumption before this workspace is unknown.

No GPU, paid resource, external dashboard, publication or deployment was used. The multi-seed representation matrix, fidelity-aligned completion guidance, internal-window editor, terminal measurements and dynamic/hardware extensions remain unrun.
