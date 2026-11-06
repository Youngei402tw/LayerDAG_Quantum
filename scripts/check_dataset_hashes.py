"""Regenerate the frozen QAI final datasets (no training) and compare content hashes.

Read-only with respect to results. Also checks train/test exact-duplicate overlap.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from qdag.invariant_benchmark import relabel
from qdag.structured_transfer import generate_structured
from qdag.schedule_benchmark import examples_hash

fam = ("qaoa", "vqe", "arithmetic")
splits = {
    "training": generate_structured(14100, 1000, (4, 7), (32, 64), fam),
    "validation": generate_structured(14101, 256, (4, 7), (32, 64), fam),
    "same": generate_structured(14102, 512, (4, 7), (32, 64), fam),
    "near_width": generate_structured(14103, 512, (8, 12), (72, 104), fam),
    "far_width": generate_structured(14104, 512, (13, 20), (96, 144), fam),
}
rng = np.random.default_rng(14105)
splits["renamed"] = [relabel(x, rng) for x in splits["same"]]
recorded = json.loads((ROOT / "experiments/qai_invariant_final_20261102/final_result.json").read_text())["dataset_hashes"]
ok = True
for k, v in splits.items():
    h = examples_hash(v); match = h == recorded[k]; ok &= match
    q = [x.n_qubits for x in v]; g = [len(x.ops) for x in v]; pos = np.concatenate([x.labels for x in v]).mean()
    print(f"{'OK ' if match else 'BAD'} {k:11s} n={len(v)} qubits {min(q)}-{max(q)} gates {min(g)}-{max(g)} pos_frac={pos:.3f}")
def key(x): return json.dumps([(o.gate, o.wires, o.params) for o in x.ops])
train_keys = {key(x) for x in splits["training"]}
for k in ("validation", "same", "near_width", "far_width"):
    print(f"    exact duplicates of training circuits in {k}: {sum(key(x) in train_keys for x in splits[k])}")
# relabeling is a consistent permutation: labels identical and gate multiset per op identical
same, ren = splits["same"], splits["renamed"]
assert all(np.array_equal(a.labels, b.labels) and [o.gate for o in a.ops] == [o.gate for o in b.ops] for a, b in zip(same, ren))
nontrivial = sum(key(a) != key(b) for a, b in zip(same, ren))
print(f"    relabeled circuits differing from original: {nontrivial}/{len(same)}; labels identical for all")
sys.exit(0 if ok else 1)
