# Frozen five-seed structured circuit transfer

The predeclared final gate passed on fresh data. Training used balanced QAOA-like, hardware-efficient VQE, and reversible linear-arithmetic circuits with 4--7 qubits and 32--64 gates. Every test split contained 512 fresh circuits. Models were parameter matched and received 4,000 updates for paired seeds 81--85.

| Split | DAG AUROC | Grid AUROC | DAG-grid AUROC | DAG F1 | Grid F1 |
|---|---:|---:|---:|---:|---:|
| Same size | 0.9677 ± 0.0011 | 0.9623 ± 0.0012 | +0.0054 | 0.9300 | 0.9212 |
| Width shift | **0.8922 ± 0.0064** | 0.8408 ± 0.0124 | **+0.0514** | **0.8378** | 0.7995 |
| Depth shift | **0.9589 ± 0.0015** | 0.9564 ± 0.0007 | +0.0025 | **0.9250** | 0.9132 |
| Width + depth | **0.8902 ± 0.0048** | 0.8495 ± 0.0131 | **+0.0407** | **0.8351** | 0.7928 |
| Held-out chemistry | 0.5571 ± 0.0127 | **0.5837 ± 0.0058** | -0.0266 | **0.6141** | 0.5994 |

Values are five-seed means ± sample standard deviation where shown. Simultaneous-shift paired AUROC differences were +0.0321, +0.0243, +0.0533, +0.0414, and +0.0525. Their mean was +0.0407, above the frozen +0.015 threshold; DAG also improved simultaneous-shift F1 by +0.0423 and width-shift AUROC by +0.0514.

The supported claim is within-family size transfer. The axis decomposition shows that circuit-width growth drives most of the observed advantage. The negative held-out chemistry result is diagnostic and demonstrates that the learned representation does not automatically generalize to a new circuit family.
