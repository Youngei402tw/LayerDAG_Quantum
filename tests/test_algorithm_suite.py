import numpy as np

from qdag.algorithm_suite import BASIS, FAMILIES, MIN_GATES, commutation_labels, compile_circuit, generate_suite
from qdag.schedule_benchmark import examples_hash


def test_suite_is_deterministic_basis_only_and_unique():
    a, names = generate_suite(7, 22, (4, 7)); b, _ = generate_suite(7, 22, (4, 7))
    assert examples_hash(a) == examples_hash(b) and names == list(FAMILIES) * 2
    assert len({examples_hash([x]) for x in a}) == len(a)
    for x in a:
        assert len(x.ops) >= MIN_GATES and len(x.labels) == len(x.ops) and 4 <= x.n_qubits <= 7
        assert {op.gate for op in x.ops} <= set(BASIS)


def test_every_family_scales_to_twenty_qubits():
    examples, names = generate_suite(8, len(FAMILIES), (20, 20))
    assert all(x.n_qubits == 20 for x in examples) and set(names) == set(FAMILIES)


def test_commutation_labels_hand_example():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2); qc.rz(0.3, 0); qc.cx(0, 1); qc.h(1)
    # RZ on the control commutes with CX; H on the target does not: longest chain is CX -> H.
    assert commutation_labels(qc).tolist() == [0.0, 1.0, 1.0]
    examples, _, compiled = generate_suite(9, 11, (4, 7), return_compiled=True)
    for x, c in zip(examples, compiled):
        labels = commutation_labels(c); assert labels.shape == x.labels.shape and labels.max() == 1.0
