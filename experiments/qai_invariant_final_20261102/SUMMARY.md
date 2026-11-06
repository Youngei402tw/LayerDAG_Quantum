# Frozen five-seed qubit-label-invariant final

The predeclared QAI gate passed on fresh data. Both encoders used qubit-index-free node features and near-matched parameter counts (80,833 DAG; 84,961 grid). Training used 1,000 balanced QAOA-like, VQE and arithmetic circuits with 4--7 qubits and 32--64 gates. Each test split contained 512 circuits. Paired seeds 111--115 received 4,000 updates.

| Split | DAG AUROC | Grid AUROC | DAG F1 | Grid F1 |
|---|---:|---:|---:|---:|
| Same size, 4--7 qubits | **0.9722 ± 0.0009** | 0.9391 ± 0.0028 | **0.9383** | 0.8919 |
| Near width, 8--12 qubits | **0.9362 ± 0.0008** | 0.8853 ± 0.0027 | **0.8749** | 0.8034 |
| Far width, 13--20 qubits | **0.9252 ± 0.0020** | 0.8807 ± 0.0045 | **0.8458** | 0.8011 |
| Relabeled same-size | **0.9722 ± 0.0009** | 0.8294 ± 0.0053 | **0.9383** | 0.8199 |

Far-width paired AUROC differences were +0.0419, +0.0419, +0.0417, +0.0482 and +0.0484; mean +0.0444. The descriptive seed-level 95% t interval is [+0.0400, +0.0488]. Far-width F1 improved by +0.0446.

The role-invariant DAG's maximum logit change under random logical-qubit relabeling was exactly 0.0, and all aggregate metrics were identical. Grid AUROC fell by 0.1097 after relabeling because canonical sequence order changes even without absolute index features.

The supported claim is that operand-role dependency edges provide exact qubit-label symmetry and improve within-family extrapolation from 4--7 to 13--20 qubits. This does not establish unseen-family generalization or practical compiler optimization.
