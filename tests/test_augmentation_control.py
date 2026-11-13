import numpy as np

from qdag.augmentation_control import EXPECTED_PARAMETERS, MODELS, train_augmented
from qdag.invariant_benchmark import relabel
from qdag.structured_transfer import generate_structured


def test_parameter_counts_match_protocol():
    for kind, cls in MODELS.items():
        assert sum(p.numel() for p in cls().parameters()) == EXPECTED_PARAMETERS[kind]


def test_relabel_preserves_labels_and_structure():
    rng = np.random.default_rng(3)
    for x in generate_structured(9, 8, (4, 7), (32, 64), ("qaoa", "vqe", "arithmetic")):
        y = relabel(x, rng)
        assert np.array_equal(x.labels, y.labels) and [o.gate for o in x.ops] == [o.gate for o in y.ops]


def test_augmented_training_runs_and_checkpoints():
    data = generate_structured(10, 20, (4, 7), (32, 64), ("qaoa", "vqe", "arithmetic"))
    for kind in MODELS:
        model, info = train_augmented(kind, data[:16], data[16:], 111, 2)
        assert info["parameters"] == EXPECTED_PARAMETERS[kind] and len(info["history"]) == 1
