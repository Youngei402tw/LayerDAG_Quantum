# Frozen development protocol: structured circuit transfer

## Question

Does the wire-dependency DAG advantage extend beyond iid random circuits to structured parametric circuit families, and is it driven by circuit width, depth, or both?

## Data

Generate deterministic QAOA-like, hardware-efficient VQE and reversible linear-arithmetic circuits using the family builders already verified by the routed-rewrite study. Do not insert rewrite templates. Development training contains 600 circuits with 4--7 qubits and 32--64 gates; validation contains 128 fresh circuits from the same range. Evaluation contains 128 circuits per split:

- same size: 4--7 qubits, 32--64 gates;
- width shift: 8--12 qubits, 32--64 gates;
- depth shift: 4--7 qubits, 72--104 gates;
- width and depth shift: 8--12 qubits, 72--104 gates;
- held-out family: chemistry-inspired circuits, 8--12 qubits, 72--104 gates.

Families are balanced by deterministic round-robin assignment. Exact depth-critical labels use the same wire-dependency dynamic program as the frozen random-circuit study.

## Models and gate

Use the frozen 87,745-parameter six-layer DAG and 88,417-parameter canonical-grid GRU, identical features, batches, optimizer, validation checkpointing, and 2,000 updates. Run one paired seed (71). Advance to a fresh five-seed final test only if DAG AUROC exceeds grid on the simultaneous width-and-depth shift. Held-out-family performance is diagnostic and is not required for advancement. No architecture or data revision follows this development result.
