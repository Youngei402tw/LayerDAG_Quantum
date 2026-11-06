# Frozen final protocol: qubit-label-invariant QAI benchmark

Development passed without architecture revision. The role-invariant DAG reached 0.9203 AUROC on 13--20-qubit circuits versus 0.8698 for the matched invariant-feature grid and had exactly zero logit change under random logical-qubit renaming. The grid's renamed AUROC fell from 0.9175 to 0.8432. No model, feature, optimizer, or data-range change follows development.

## Fresh final data

Use new base seeds 14100--14105. Train on 1,000 balanced QAOA-like, VQE and reversible-arithmetic circuits with 4--7 qubits and 32--64 gates. Validate on 256 fresh circuits. Evaluate on 512 circuits per split:

- same size: 4--7 qubits, 32--64 gates;
- near width: 8--12 qubits, 72--104 gates;
- far width: 13--20 qubits, 96--144 gates;
- randomly relabeled copy of the same-size test.

Train paired seeds 111--115 for 4,000 updates. The DAG has 80,833 parameters and grid 84,961. Both receive the same qubit-index-free node features; the DAG alone receives operand-role edge relations inherent to its representation.

## Confirmatory gate

Pass only if:

1. DAG far-width AUROC exceeds grid for all five paired seeds;
2. mean far-width AUROC improvement is at least +0.03;
3. mean far-width F1 exceeds grid;
4. maximum DAG logit difference under qubit relabeling is at most 1e-6;
5. DAG same-size metrics are unchanged by relabeling to numerical precision.

Report grid degradation under relabeling as a diagnostic. The claim concerns qubit-label invariance and within-family width extrapolation, not unseen-family transfer or compiler optimization.
