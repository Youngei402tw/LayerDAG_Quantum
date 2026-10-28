# Frozen circuit-size transfer result

The predeclared positive gate passed on fresh data and three paired training
seeds. Models received identical gate/wire-role inputs and 6,000 updates.

| Encoding | Parameters | Same-size AUROC | Reordered AUROC | Larger-circuit AUROC | Larger F1 |
|---|---:|---:|---:|---:|---:|
| Wire-dependency DAG MPNN | 87,745 | 0.8556 ± 0.0042 | 0.8556 ± 0.0042 | **0.8078 ± 0.0015** | **0.6341 ± 0.0033** |
| Canonical earliest-layer grid GRU | 88,417 | **0.9406 ± 0.0008** | **0.9406 ± 0.0008** | 0.7861 ± 0.0035 | 0.6055 ± 0.0037 |

Values are seed means ± sample standard deviations for seeds 51, 52, and 53.
The paired larger-circuit AUROC differences were +0.0202, +0.0177, and
+0.0274; mean +0.0217. Mean larger-circuit F1 improved by +0.0286.

The supported claim is that explicit wire-dependency DAG message passing
generalizes better than a parameter-matched canonical grid when predicting
depth-critical gates under a shift from 4–8 qubit, 24–56 gate circuits to
9–12 qubit, 64–96 gate circuits. The grid is clearly better in-distribution.
This result concerns compiler critical-path reasoning, not state-preparation
fidelity or universal DAG superiority.
