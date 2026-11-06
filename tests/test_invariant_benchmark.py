import numpy as np

from qdag.invariant_benchmark import InvariantTransformer, invariant_node_features, relabel, role_edges
from qdag.structured_transfer import generate_structured


def test_role_features_ignore_absolute_qubit_labels():
    example = generate_structured(9, 1, (5, 5), (24, 24), ("qaoa",))[0]
    changed = relabel(example, np.random.default_rng(10))
    assert np.array_equal(invariant_node_features(example), invariant_node_features(changed))
    edge_a, attr_a = role_edges(example)
    edge_b, attr_b = role_edges(changed)
    assert np.array_equal(edge_a, edge_b)
    assert np.array_equal(attr_a, attr_b)
    assert np.array_equal(example.labels, changed.labels)


def test_invariant_transformer_parameter_scale_is_matched():
    model = InvariantTransformer()
    assert 75_000 <= sum(p.numel() for p in model.parameters()) <= 95_000
