import numpy as np
import torch

from qdag.core import Op, simulate
from qdag.learning import (
    CATALOG, MASK, FactorizedFrontierModel, FrontierAwareMPNN, NativeMPNN, action_components,
    angle_bin, bin_angle, canonical_layers, decode_layer, dense_message_oracle, damaged_prefix,
    density_features, encode_layer, legal_mask,
)


def test_action_roundtrip_preserves_continuous_angles():
    layers = [
        [Op("ry", (0,), (0.317,)), Op("rz", (1,), (-2.311,))],
        [Op("cx", (1, 0))],
        [Op("rz", (0,), (3.08,))],
    ]
    rebuilt = []
    for layer in layers:
        action, residual = encode_layer(layer)
        rebuilt.extend(decode_layer(action, residual))
    expected = [op for layer in layers for op in layer]
    np.testing.assert_allclose(simulate(2, expected), simulate(2, rebuilt), atol=1e-13)
    np.testing.assert_allclose(
        [op.params[0] for op in expected if op.params],
        [op.params[0] for op in rebuilt if op.params],
        atol=1e-14,
    )


def test_periodic_bin_roundtrip_and_global_phase_features():
    for theta in np.linspace(-20, 20, 101):
        b, r = angle_bin(float(theta))
        assert abs(np.angle(np.exp(1j * (bin_angle(b, r) - theta)))) < 1e-12
    state = np.array([1, 1j, -0.3, 0.2j], complex)
    state /= np.linalg.norm(state)
    np.testing.assert_allclose(density_features(state), density_features(state * np.exp(0.731j)), atol=1e-7)


def test_hard_legality_masks_stop_and_bounds():
    mask = legal_mask((0, 0, 0), torch.device("cpu"))
    assert mask[0] and int(mask.sum()) == 1
    mask = legal_mask((1, 1, 0), torch.device("cpu"))
    for i, action in enumerate(CATALOG):
        if action.kind == "cx":
            assert not mask[i]
        if action.kind == "rot" and action.left_gate and action.right_gate:
            assert not mask[i]


def test_native_index_add_matches_dense_oracle():
    torch.manual_seed(4)
    encoder = NativeMPNN(width=8)
    h = torch.randn(4, 8)
    edge = torch.tensor([[0, 1, 0, 2], [1, 2, 3, 3]])
    attr = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
    index_add = torch.zeros(4, 8)
    index_add.index_add_(0, edge[1], encoder.forward_layers[0](torch.cat([h[edge[0]], attr], 1)))
    torch.testing.assert_close(index_add, dense_message_oracle(h, edge, attr, encoder.forward_layers[0]))
    reverse = torch.zeros(4, 8)
    reverse.index_add_(0, edge[0], encoder.backward_layers[0](torch.cat([h[edge[1]], attr], 1)))
    torch.testing.assert_close(reverse, dense_message_oracle(h, edge, attr, encoder.backward_layers[0], reverse=True))


def test_damaged_prefix_never_qualifies_unchanged():
    witnesses = [
        [[Op("ry", (0,), (0.01,))], [Op("rz", (0,), (0.02,))]],
        [[Op("ry", (0,), (1.1,))], [Op("cx", (0, 1))], [Op("rz", (1,), (-0.4,))]],
    ]
    for layers in witnesses:
        target = simulate(2, [op for layer in layers for op in layer])
        prefix = damaged_prefix(target, layers)
        fidelity = abs(np.vdot(target, simulate(2, [op for layer in prefix for op in layer]))) ** 2
        assert fidelity < 0.99


def test_canonical_layers_match_physical_depth_and_merge_disjoint_ops():
    ops = [Op("ry", (0,), (0.2,)), Op("rz", (1,), (-0.3,)), Op("cx", (0, 1)), Op("rz", (0,), (0.7,))]
    layers = canonical_layers(ops)
    assert len(layers) == 3
    assert layers[0] == ops[:2]
    assert [op for layer in layers for op in layer] == ops


def test_factorized_catalog_components_reconstruct_every_action():
    assert len({action_components(action) for action in CATALOG}) == len(CATALOG)
    assert action_components(CATALOG[0]) == (0, 0, 0)
    model = FactorizedFrontierModel("dag", width=8)
    target = torch.zeros((1, 32))
    prefix = torch.zeros((1, 32))
    remaining = torch.ones((1, 3))
    logits, residual = model([[]], target, prefix, remaining, torch.tensor([MASK]), torch.tensor([8]))
    assert logits.shape == (1, len(CATALOG))
    assert residual.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_frontier_aware_encoder_distinguishes_wire_frontiers():
    torch.manual_seed(5)
    encoder = FrontierAwareMPNN(width=8)
    a = [[Op("ry", (0,), (0.2,))], [Op("rz", (1,), (0.3,))]]
    b = [[Op("rz", (1,), (0.3,))], [Op("ry", (0,), (0.2,))]]
    outputs = encoder([a, b])
    assert not torch.allclose(outputs[0], outputs[1])
