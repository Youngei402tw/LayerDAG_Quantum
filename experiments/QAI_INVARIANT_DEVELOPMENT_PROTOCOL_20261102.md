# Frozen QAI development protocol: qubit-label-invariant size transfer

## Motivation

The confirmed encoder uses absolute wire-membership and wire-index edge features with a fixed 12-qubit feature width. This limits extrapolation and does not guarantee invariance to arbitrary logical-qubit renaming. For a QAI representation-learning claim, test whether relational operand-role features remove that limitation.

## Encoders

Both models receive only gate type, angle sine/cosine and arity at each node. Neither receives an absolute qubit index. DAG edge features encode the role of the shared wire at the source and destination operation: single-qubit, CX control, or CX target. The canonical-grid control uses identical node features and deterministic earliest-layer ordering. Select widths so parameter counts differ by less than 10%.

## Data

Use balanced QAOA-like, VQE and reversible arithmetic circuits without rewrite insertions. Development training contains 600 circuits with 4--7 qubits and 32--64 gates; validation contains 128 fresh circuits. Test splits contain 128 circuits each:

- same size: 4--7 qubits, 32--64 gates;
- near width shift: 8--12 qubits, 72--104 gates;
- far width shift: 13--20 qubits, 96--144 gates;
- randomly relabeled copy of the same-size test.

Train one paired seed (101) for 2,000 updates. Advance only if DAG far-width AUROC exceeds grid and DAG logits are invariant to qubit relabeling within 1e-6. No final data or architecture tuning follows a failed development gate.
