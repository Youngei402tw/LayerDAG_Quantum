import numpy as np
import torch

from qdag.core import Op
from qdag.schedule_benchmark import (
    CircuitExample, DagCriticalModel, canonical_order, critical_labels, dependency_edges,
    random_topological_permutation, reorder_example,
)


def test_critical_labels_follow_wire_dependency_longest_path():
    ops = [Op("h", (0,)), Op("h", (1,)), Op("cx", (0, 1)), Op("rz", (2,), (0.2,)), Op("ry", (0,), (0.3,))]
    labels = critical_labels(3, ops)
    np.testing.assert_array_equal(labels, [1, 1, 1, 0, 1])


def test_topological_reordering_preserves_dependency_graph_and_labels():
    ops = [Op("h", (0,)), Op("ry", (1,), (0.2,)), Op("rz", (2,), (0.3,)), Op("cx", (0, 1)), Op("cx", (2, 3)), Op("rz", (3,), (0.4,))]
    example = CircuitExample(4, ops, critical_labels(4, ops))
    changed = reorder_example(example, 7)
    assert sorted(map(tuple, dependency_edges(4, example.ops)[0].T)) != []
    assert int(changed.labels.sum()) == int(example.labels.sum())
    np.testing.assert_array_equal(changed.labels, critical_labels(4, changed.ops))
    original_canonical = [example.ops[i] for i in canonical_order(example)]
    changed_canonical = [changed.ops[i] for i in canonical_order(changed)]
    assert original_canonical == changed_canonical


def test_dag_logits_are_equivariant_to_legal_topological_reordering():
    torch.manual_seed(3)
    ops = [Op("h", (0,)), Op("ry", (1,), (0.2,)), Op("rz", (2,), (0.3,)),
           Op("cx", (0, 1)), Op("cx", (2, 3)), Op("rz", (3,), (0.4,))]
    example = CircuitExample(4, ops, critical_labels(4, ops))
    changed = reorder_example(example, 11)
    order = random_topological_permutation(np.random.default_rng(11), 4, ops)
    model = DagCriticalModel(width=8, layers=2).eval()
    with torch.no_grad():
        original = model([example])[0]
        restored = torch.empty_like(original)
        restored[torch.tensor(order)] = model([changed])[0]
    torch.testing.assert_close(original, restored, atol=1e-6, rtol=0)
