import numpy as np

from qdag.routed_rewrite import generate_family_examples, remove_candidate
from qdag.core import simulate


def test_generated_rewrite_templates_are_exact():
    examples = generate_family_examples(7, 4, (4, 4), (16, 18), ("qaoa", "vqe", "arithmetic", "chemistry"))
    for example in examples:
        reference = simulate(example.circuit.n_qubits, example.circuit.ops)
        assert {c.template for c in example.candidates} == {"bound_zero_rotation", "inverse_rotation_pair", "self_inverse_cx_pair"}
        for candidate in example.candidates:
            edited = simulate(example.circuit.n_qubits, remove_candidate(example, candidate))
            assert abs(np.vdot(reference, edited)) ** 2 > 1 - 1e-10
