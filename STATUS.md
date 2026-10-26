# Actual status

Final execution status: **SCIENTIFIC GATE FAILED**.

Implemented and verified:

- Independent complex128 statevector simulator cross-checked with Qiskit for ordered X/H/SX/RY/RZ/CX/SWAP/CSWAP/CCX semantics, continuous angles and global phase.
- Typed version-1 logical constraint evaluator with explicit unsupported, inconsistent, qualified, exhausted and incomplete outcomes; exact/min/max operation counts, total gates, dependency depth, two-qubit depth, ordered operands/connectivity and zero ancillas.
- Expanded reset-free Qiskit StatePreparation seeds, level-0/level-3 compiler controls, bounded classical deletion/RZ refinement and deterministic gate-set repair.
- Canonical earliest-layer RY/RZ/CX witnesses, phase-invariant target grouping, train/validation duplicate rejection, target/prefix density features, remaining budgets, layer-zero STOP and hard legal-frontier masks.
- Portable native-PyTorch LayerDAG-derived encoder with labelled wire dependencies and dense-oracle aggregation tests; matched canonical-grid control; absorbing-mask categorical frontier denoising; 16-bin periodic angles with learned bounded residuals.
- Checkpointed CPU training, immutable per-request sampling streams, K=1/K=32 tiny evaluation and paired correct/permuted/null conditioning diagnostics with target-level bootstrap intervals.

Evidence:

- 19 tests pass. All 240 compiler-pilot rows replay against Qiskit; maximum original fidelity discrepancy is 1.11e-15.
- Corrected tiny learning gate: DAG repaired 31/32 damaged suffixes and grid 31/32 at K=32; no damaged prefix qualified unchanged and every sampled circuit met its budgets. From scratch, DAG reached 32/32 and grid 31/32 at K=32.
- Held-out 32-target condition check after 128-target development training: grid passed (mean correct-minus-permuted +0.1300, bootstrap 95% interval [+0.0133, +0.2407]); width-64 DAG failed (+0.0327, [-0.0704, +0.1304]). The final width-80 DAG diagnosis also failed (+0.1018, [-0.0073, +0.2118]).
- The first learned DAG run is retained under `experiments/tiny_learning_20261021_invalid_noncanonical` and is invalidated because declared layers did not always equal earliest physical dependency layers.

Consequences:

- The implementation and tiny training-target repair gate work, but held-out target dependence was not established for the DAG model after bounded diagnosis. There is no evidence that the DAG representation beats the grid representation.
- Guidance, the 2x2 guide/representation matrix, multi-seed research matrix, internal-window learning, 3/4-qubit learned scaling, hardware/noise scopes, parser, readout and dynamic circuits were not run. The handoff requires stopping before these stages when the conditioning gate fails.
- Checkpoints contain model, optimizer and RNG state, but the current CLI does not resume an interrupted training loop from its exact data position; completed checkpoints support evaluation replay. Faithful mid-training resume remains an engineering gap.
- PyTorch 2.8.0+cpu is used; DGL runtime parity with upstream is unverified. Upstream remains clean at `bc22fed6f112c3eef2e991551bb0677038df3467`.

Workspace-measured compute-job wall time is 2,274.45 seconds (37.91 minutes), including the invalidated run and evaluation. Failed setup/replay attempts and engineering time were not fully timed. Historic consumption before this workspace remains unknown, so this ledger does not certify the cumulative six-hour allowance. No GPU, paid resource, publication or background job was used.

The exact next bounded experiment, if a future execution authorizes a new research iteration, is a fresh predeclared multi-seed comparison after redesigning DAG prefix/frontier features. It must use new validation/test targets; it should not tune further on the validation set already inspected here.
