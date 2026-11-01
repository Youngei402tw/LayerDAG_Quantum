from qdag.compiler_utility import IdentityRewriteExample, candidate_depth_reductions
from qdag.core import Op
from qdag.schedule_benchmark import CircuitExample, critical_labels


def test_identity_removal_can_reduce_dependency_depth():
    ops = [Op("h", (0,)), Op("rz", (0,), (0.0,)), Op("cx", (0, 1))]
    circuit = CircuitExample(2, ops, critical_labels(2, ops))
    reductions = candidate_depth_reductions(IdentityRewriteExample(circuit, [1]))
    assert reductions.tolist() == [1.0]
