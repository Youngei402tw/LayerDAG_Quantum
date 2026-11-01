import numpy as np

from qdag.core import Op
from qdag.rewrite_benchmark import FIDELITY_MIN, deletion_fidelities, heuristic_scores
from qdag.schedule_benchmark import CircuitExample


def test_deletion_fidelity_detects_identity_rotation():
    ops = [Op("ry", (0,), (0.0,)), Op("h", (1,)), Op("cx", (1, 0))]
    scores = deletion_fidelities(2, ops)
    assert scores[0] >= FIDELITY_MIN
    assert scores[1] < FIDELITY_MIN


def test_angle_heuristic_prefers_small_rotations():
    ops = [Op("ry", (0,), (0.01,)), Op("rz", (1,), (1.2,)), Op("cx", (0, 1))]
    example = CircuitExample(2, ops, np.zeros(3, np.float32))
    assert int(np.argmax(heuristic_scores(example))) == 0
