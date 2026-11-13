import numpy as np
import torch

from qdag.connectivity_controls import EXPECTED_PARAMETERS, MODELS, pointer_features
from qdag.core import Op
from qdag.invariant_benchmark import relabel
from qdag.schedule_benchmark import CircuitExample, canonical_order, critical_labels
from qdag.structured_transfer import generate_structured


def example(ops, n=3):
    return CircuitExample(n, ops, critical_labels(n, ops))


def test_parameter_counts_match_protocol():
    for kind, cls in MODELS.items():
        assert sum(p.numel() for p in cls().parameters()) == EXPECTED_PARAMETERS[kind]


def test_pointer_features_hand_example():
    # q0: H(0) -> CX(0,1) control ; q1: CX target -> RZ(1)
    x = example([Op("h", (0,), ()), Op("cx", (0, 1), ()), Op("rz", (1,), (0.3,))], n=2)
    order = canonical_order(x); assert order == [0, 1, 2]
    f = pointer_features(x)
    assert f.shape == (3, 33)
    h, cx, rz = f
    # H: role single, no predecessor, successor is CX in control role at distance 1
    assert list(h[7:10]) == [1, 0, 0] and h[10] == 0 and h[15] == 1 and list(h[16:19]) == [0, 1, 0]
    assert np.isclose(h[19], np.log1p(1) / np.log1p(192)) and not h[20:].any()
    # CX slot 0 (control): predecessor H single; slot 1 (target): no predecessor, successor RZ single
    assert list(cx[7:10]) == [0, 1, 0] and cx[10] == 1 and list(cx[11:14]) == [1, 0, 0] and cx[15] == 0
    assert list(cx[20:23]) == [0, 0, 1] and cx[23] == 0 and cx[28] == 1 and list(cx[29:32]) == [1, 0, 0]
    # RZ: predecessor is CX in target role
    assert rz[10] == 1 and list(rz[11:14]) == [0, 0, 1] and not rz[20:].any()


def test_pointer_roles_and_existence_are_relabeling_invariant():
    rng = np.random.default_rng(0)
    for x in generate_structured(5, 12, (4, 7), (32, 64), ("qaoa", "vqe", "arithmetic")):
        y = relabel(x, rng)
        fx = pointer_features(x)[np.argsort(canonical_order(x))]
        fy = pointer_features(y)[np.argsort(canonical_order(y))]
        keep = [i for i in range(33) if i not in (14, 19, 27, 32)]  # distance columns depend on tie-break order
        assert np.array_equal(fx[:, keep], fy[:, keep])


def test_role_ablated_dag_is_exactly_relabeling_invariant():
    torch.manual_seed(0); model = MODELS["role_ablated_dag"]().eval(); rng = np.random.default_rng(1)
    for x in generate_structured(6, 6, (4, 7), (32, 64), ("qaoa", "vqe", "arithmetic")):
        with torch.no_grad():
            assert torch.equal(model([x])[0], model([relabel(x, rng)])[0])


def test_pointer_models_return_logits_in_original_op_order():
    xs = generate_structured(7, 3, (4, 7), (32, 64), ("qaoa",))
    for kind in ("pointer_grid", "pointer_transformer"):
        out = MODELS[kind]()(xs)
        assert [len(o) for o in out] == [len(x.ops) for x in xs]
