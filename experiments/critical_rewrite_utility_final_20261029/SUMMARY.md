# DAG-guided safe rewrite prioritization

The frozen fresh-data final test produced a consistent DAG advantage in all
three paired seeds, but the mean improvement did not reach the predeclared
five-percentage-point confirmatory threshold.

| Ranking | Oracle depth recovered | Mean depth reduction | Oracle-best choice |
|---|---:|---:|---:|
| Wire-dependency DAG | **0.5194 ± 0.0058** | **0.4017 ± 0.0045** | **0.6283 ± 0.0045** |
| Canonical grid | 0.4815 ± 0.0119 | 0.3724 ± 0.0092 | 0.5990 ± 0.0092 |
| Random | 0.1793 | 0.1387 | 0.3652 |

The paired DAG-minus-grid recovery differences were +0.0455, +0.0404, and
+0.0278. Their mean, +0.0379, is positive but below the frozen +0.05 gate.
Therefore the confirmatory claim failed even though every measured seed favored
DAG.

All candidate rewrites were injected RZ(0) identities and therefore exactly
fidelity preserving. This establishes encouraging downstream scheduling
utility under a controlled size shift. A publishable compiler claim still
requires natural rewrite opportunities from real circuit families and
comparison with production compiler heuristics.
