# Frozen final protocol: DAG-guided safe depth reduction

The development pilot showed that a wire-dependency DAG ranking recovered more
of the exact best depth reduction than a parameter-matched canonical grid when
choosing among several fidelity-preserving identity removals on larger circuits.

Freeze the width-64 six-layer DAG MPNN and width-96 canonical-grid GRU. Train
both for 3,000 updates on the same fresh 1,000 random 4–8-qubit circuits, select
checkpoints using 256 fresh validation circuits, and evaluate on 512 fresh
9–12-qubit circuits with one explicit RZ(0) rewrite candidate per qubit. Use
training seeds 91, 92, and 93. All candidates are exactly fidelity preserving.
The exact oracle tests every candidate's dependency-depth reduction; each
learned model selects one candidate without oracle feedback.

The final gate passes only if:

1. DAG oracle-reduction recovery exceeds canonical grid in every seed;
2. the mean recovery advantage is at least +0.05;
3. DAG mean realized depth reduction and exact-oracle-choice rate both exceed
   canonical grid;
4. both models use their frozen parameterization and equal update budgets.

Report every seed, means and sample standard deviations, random selection, and
the oracle upper bound. The supported claim is limited to prioritizing known
safe rewrites under circuit-size shift. Identity candidates are synthetic, so
validation on natural compiler rewrite opportunities remains future work.
