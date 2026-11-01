# Critical rewrite utility pilot

The single-gate fidelity pilot was saturated by a smallest-angle heuristic.
This follow-up isolates downstream scheduling utility. Train the frozen DAG and
canonical-grid criticality encoders on 4–8-qubit random circuits. Evaluate on
256 fresh 9–12-qubit circuits containing several explicit RZ(0) identities.
Every candidate is exactly fidelity preserving; the decision is which removal
reduces global dependency depth the most.

For one development seed and 3,000 updates per model, compare top-1 realized
depth reduction, fraction of circuits with any reduction, fraction matching
the exact best candidate, and fraction of oracle depth reduction recovered.
Include random candidate selection and an exact exhaustive oracle. This is a
development pilot, not a final claim. It passes when DAG exceeds canonical grid
on mean depth reduction and recovered-oracle fraction.
