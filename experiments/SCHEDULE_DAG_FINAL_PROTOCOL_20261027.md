# Frozen final protocol: circuit-DAG size transfer

Development compared a six-layer wire-dependency MPNN, a full-depth recurrent
DAG, a raw bidirectional gate sequence, a canonical earliest-layer sequence,
and a token Transformer. The recurrent DAG was slower and weaker. The canonical
grid became the strongest in-distribution control after extended training, but
the fixed-depth MPNN retained better transfer to larger circuits.

Freeze the six-layer width-64 DAG MPNN (87,745 parameters) and width-96
canonical-grid bidirectional GRU (88,417 parameters). Give both 6,000 updates,
identical minibatches in distribution, the same optimizer, and validation-only
checkpoint selection. Use fresh deterministic data: 1,000 training and 256
validation circuits with 4–8 qubits and 24–56 gates; 512 final in-distribution
circuits; their legally reordered equivalents; and 512 final size-shifted
circuits with 9–12 qubits and 64–96 gates. Train seeds 51, 52, and 53.

The positive size-transfer claim passes only if:

1. DAG larger-circuit AUROC exceeds canonical grid in all three seeds;
2. the seed-mean AUROC difference is at least +0.015;
3. DAG seed-mean larger-circuit F1 exceeds canonical grid;
4. DAG reordering logits remain invariant within 1e-5.

Report canonical, reordered, and larger-circuit AUROC/F1/accuracy, parameter
counts, updates, and wall time. This establishes an advantage for predicting
wire-dependency critical paths under circuit-size shift. It does not establish
better state preparation or universal GNN superiority.
