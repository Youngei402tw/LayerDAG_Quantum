import json
import pickle

import numpy as np
import pytest
import torch

from qdag import algorithm_suite_run as run
from qdag.invariant_benchmark import relabel
from qdag.schedule_benchmark import examples_hash

TINY = (("training", 90300, 22, (4, 6)), ("validation", 90301, 11, (4, 6)), ("same", 90302, 11, (4, 6)),
        ("near_width", 90303, 11, (8, 9)), ("far_width", 90304, 11, (13, 14)))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    folder = tmp_path_factory.mktemp("suite"); data = run.build_data(TINY, {"same": 90305, "far_width": 90306}); record = run.manifest(data)
    (folder / "data_manifest.json").write_text(json.dumps(record))
    with open(folder / "cache.pkl", "wb") as handle: pickle.dump({"hashes": record["hashes"], "families": data["families"], "A": data["A"], "B": data["B"]}, handle)
    return folder, data, record


def test_permute_example_matches_frozen_relabel_stream(tiny):
    _, data, _ = tiny; a = np.random.default_rng(5); b = np.random.default_rng(5)
    for x in data["A"]["same"]:
        frozen = relabel(x, a); mine = run.permute_example(x, b.permutation(x.n_qubits))
        assert frozen.ops == mine.ops and np.array_equal(frozen.labels, mine.labels)


def test_data_splits_are_disjoint_and_variants_keep_labels(tiny):
    _, data, record = tiny; keys = [examples_hash([x]) for s in ("training", "validation", "same", "near_width", "far_width") for x in data["A"][s]]
    assert len(set(keys)) == len(keys)
    for variant, base in run.VARIANTS.items():
        assert record["renaming_checks"][variant]["labels_recomputed_and_equal"]
        for phase in ("A", "B"):
            assert all(np.array_equal(x.labels, y.labels) for x, y in zip(data[phase][variant], data[phase][base]))
    assert all(x.ops is y.ops for x, y in zip(data["A"]["training"], data["B"]["training"]))  # phases share circuits
    assert record["hashes"]["A"]["training"] != record["hashes"]["B"]["training"]  # and differ in labels


def test_auroc_handles_ties_and_single_class():
    assert run.auroc(np.array([0., 1., 1., 0.]), np.array([.1, .9, .8, .2])) == 1.0
    assert run.auroc(np.array([0., 1.]), np.array([.5, .5])) == 0.5
    assert run.auroc(np.array([1., 1.]), np.array([.1, .2])) is None


def test_split_metrics_macro_excludes_single_class_family(tiny):
    _, data, _ = tiny; examples = data["B"]["same"]; families = data["families"]["same"]
    perfect = [np.where(x.labels == 1, 5., -5.) for x in examples]; m = run.split_metrics(perfect, examples, families)
    assert m["auc"] == 1.0 and m["macro_auc"] == 1.0 and m["f1"] == 1.0 and m["accuracy"] == 1.0
    assert "oracle" in m["macro_excluded_single_class"] and m["macro_families"] == len(set(families)) - len(m["macro_excluded_single_class"])


def test_units_aggregate_and_dag_is_exactly_invariant(tiny, monkeypatch):
    folder, _, _ = tiny
    for kind in run.PHASE_KINDS["B"]:
        for seed in run.SEEDS:
            result = run.run_unit(folder, folder / "cache.pkl", "B", kind, seed, updates=1)
            if kind in ("dag", "role_ablated_dag"):
                assert all(v["max_logit_change"] == 0.0 for v in result["pair_changes"].values())
    with pytest.raises(SystemExit): run.run_unit(folder, folder / "cache.pkl", "B", "dag", 111, updates=1)  # never rerun
    out = run.aggregate(folder, "B")
    assert set(out["readings"]) == {"R1_connectivity", "R2_dag_vs_pointer", "R3_augmentation", "R4_dag_invariance", "R4_dag_max_logit_change", "roles"}
    assert out["readings"]["R4_dag_invariance"] == "exact_within_1e-6"
    assert len(out["comparisons_far_macro_auc"]["dag_minus_grid"]["differences"]) == 5


def test_reading_bands():
    def comps(dg, pg, dp, da, dr): return {k: run.paired(np.asarray(v), np.zeros(5)) for k, v in
        {"dag_minus_grid": dg, "pointer_grid_minus_grid": pg, "dag_minus_pointer_grid": dp, "dag_minus_aug_pointer_grid": da, "dag_minus_role_ablated_dag": dr}.items()}
    def agg(gs, gf, worst=0.0): return {"aug_pointer_grid": {"gaps": {"same_relabeled": {"mean": gs}, "far_relabeled": {"mean": gf}}},
                                        "dag": {"pair_changes": {v: {"max_logit_change": worst} for v in run.VARIANTS}}}
    r = run.readings("B", comps([.05] * 5, [.04] * 5, [.001, -.002, .003, 0, .001], [.012] * 5, [.02] * 5), agg(.003, -.002))
    assert r == {"R1_connectivity": "replicated", "R2_dag_vs_pointer": "tie", "R3_augmentation": "intermediate", "R4_dag_invariance": "exact_within_1e-6",
                 "R4_dag_max_logit_change": 0.0, "roles": "roles_contribute"}
    r = run.readings("A", comps([.05] * 5, [.005] * 5, [-.03] * 5, [.03] * 5, [0] * 5), agg(.001, .001, worst=1e-3))
    assert r == {"R1_connectivity": "not_replicated", "R2_dag_vs_pointer": "sequence_advantage", "R3_augmentation": "insufficient",
                 "R4_dag_invariance": "FAILED", "R4_dag_max_logit_change": 1e-3}
    r = run.readings("B", comps([.02, .03, .04, .02, .03], [.05] * 5, [.03] * 5, [.004] * 5, [.004] * 5), agg(.004, .005))
    assert (r["R1_connectivity"], r["R2_dag_vs_pointer"], r["R3_augmentation"], r["roles"]) == ("partial", "graph_advantage", "suffices", "roles_not_needed")
