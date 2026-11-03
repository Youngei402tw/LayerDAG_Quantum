# Frozen final protocol: structured circuit size transfer

Development used one paired seed and passed its predeclared simultaneous width-and-depth gate. It showed DAG-minus-grid AUROC of +0.0324 for width shift, +0.0137 for depth shift, +0.0300 for simultaneous shift, and -0.0271 on a held-out chemistry-inspired family. No architecture, feature, optimizer, or family generator changes follow that result.

## Fresh data and training

Generate fresh deterministic data with base seed 12100. Training contains 1,000 balanced QAOA-like, hardware-efficient VQE, and reversible linear-arithmetic circuits with 4--7 qubits and 32--64 gates. Validation contains 256 fresh circuits from the same distribution. Each final split contains 512 fresh circuits:

- same size: 4--7 qubits, 32--64 gates;
- width shift: 8--12 qubits, 32--64 gates;
- depth shift: 4--7 qubits, 72--104 gates;
- simultaneous shift: 8--12 qubits, 72--104 gates;
- held-out chemistry family: 8--12 qubits, 72--104 gates.

Train paired seeds 81--85. Compare the frozen width-64 six-layer DAG (87,745 parameters) and width-96 canonical-grid GRU (88,417 parameters) with identical features, minibatches, AdamW optimizer, validation-only checkpoint selection, and 4,000 updates.

## Confirmatory gate

The structured size-transfer claim passes only if:

1. DAG simultaneous-shift AUROC exceeds grid for all five paired seeds;
2. the five-seed mean simultaneous-shift AUROC difference is at least +0.015;
3. DAG mean simultaneous-shift F1 exceeds grid;
4. DAG mean width-shift AUROC exceeds grid.

Same-size, depth-only and held-out-chemistry results are reported as diagnostics and cannot rescue a failed gate. The claim concerns scale transfer within the three training families. It does not claim unseen-family generalization.
