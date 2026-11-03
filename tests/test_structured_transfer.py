from qdag.structured_transfer import generate_structured


def test_structured_families_have_exact_requested_shape_and_labels():
    examples = generate_structured(5, 8, (4, 6), (20, 24), ("qaoa", "vqe", "arithmetic", "chemistry"))
    assert len(examples) == 8
    for example in examples:
        assert 4 <= example.n_qubits <= 6
        assert 20 <= len(example.ops) <= 24
        assert len(example.labels) == len(example.ops)
        assert set(example.labels.tolist()) <= {0.0, 1.0}
        assert example.labels.sum() > 0
