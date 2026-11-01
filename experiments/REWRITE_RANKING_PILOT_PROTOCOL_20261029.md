# Fidelity-preserving rewrite-ranking pilot

This pilot tests whether the observed DAG size-transfer advantage produces a
downstream compiler benefit. For each RY/RZ/H/CX state-preparation circuit, the
target is its original output state. A gate deletion is safe when the complete
edited circuit has fidelity at least 0.999 to that target. Labels are produced
by independent full statevector simulation; models do not receive fidelity or
statevector features.

Generate 600 development-training circuits with 4–6 qubits and 20–40 gates,
128 validation circuits from the same range, and 256 size-shifted evaluation
circuits with 7–9 qubits and 40–64 gates. Include ordinary rotations, small
rotations and structured circuits; do not balance labels by changing the test
set. Compare the previously selected width-64 six-layer DAG MPNN against the
width-96 canonical-grid GRU at 3,000 updates, plus a smallest-angle heuristic
and random ranking.

For verification budgets 1, 4 and 8, rank all single-gate deletions and check
candidates in order until a safe deletion is found. Report success rate,
verification calls, fidelity, and the oracle fraction of circuits containing a
safe deletion. Development output selects whether this task is viable; it is
not a final paper claim. A positive pilot requires DAG to exceed grid at budget
4 and beat the angle heuristic without receiving more oracle calls.
