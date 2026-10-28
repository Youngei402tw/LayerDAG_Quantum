# DAG advantage exploration plan

The goal is a defensible positive claim about where circuit DAG encodings help,
not repeated tuning on a final state-preparation test.

## Development tracks

1. **Scheduling invariance and critical-path reasoning.** Predict which gates
   lie on a maximum wire-dependency path. Compare a wire-edge DAG MPNN against
   a matched bidirectional token-sequence encoder. Measure canonical accuracy,
   legal-topological-reordering robustness, and transfer to larger circuits.
2. **Constrained completion.** If track 1 validates the representation bias,
   reuse its encoder in state-preparation suffix completion and compare against
   a factorized canonical-grid model at matched parameters and sampling cost.
3. **Variable-size transfer.** Train on smaller qubit counts and test larger
   counts using shared gate and wire-role features.
4. **Fidelity-aligned search.** Compare a completion-value-guided beam against
   ordinary sampling and best-of-K reranking at matched simulator calls.

## Evidence firewall

Architecture and task exploration use development seeds only. A candidate DAG
advantage must pass all of these before a final run:

- higher development AUROC and F1 than the sequence control;
- no loss under legal topological reserialization beyond numerical tolerance
  for the DAG logits;
- a larger advantage on reordered or size-shifted circuits than in-distribution;
- matched input information, training examples, updates, and parameter scale.

After passing, freeze models and hyperparameters and generate new train,
validation, and test circuits. Run at least three training seeds and report
target-level and seed-level uncertainty. Negative tracks remain in the record.

The critical-path task is quantum-compiler relevant but does not by itself
establish better state preparation. Any paper must state that boundary.
