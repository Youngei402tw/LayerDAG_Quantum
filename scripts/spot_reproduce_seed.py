"""Spot-reproduce one paired seed of the frozen QAI final (DAG + grid) into a NEW directory.

Usage: python scripts/spot_reproduce_seed.py SEED OUTPUT_DIR
Never writes into the frozen result directory.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from qdag.invariant_benchmark import train, evaluate, relabel
from qdag.structured_transfer import generate_structured
seed = int(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=False)
fam = ("qaoa", "vqe", "arithmetic")
tr = generate_structured(14100, 1000, (4, 7), (32, 64), fam); va = generate_structured(14101, 256, (4, 7), (32, 64), fam)
same = generate_structured(14102, 512, (4, 7), (32, 64), fam); near = generate_structured(14103, 512, (8, 12), (72, 104), fam)
far = generate_structured(14104, 512, (13, 20), (96, 144), fam); rng = np.random.default_rng(14105); ren = [relabel(x, rng) for x in same]
rec = json.loads((ROOT / "experiments/qai_invariant_final_20261102/final_result.json").read_text())
rec_seed = next(r for r in rec["seed_results"] if r["seed"] == seed)
report = {"seed": seed, "torch": torch.__version__, "models": {}}
for kind in ("dag", "grid"):
    t0 = time.monotonic(); model, info = train(kind, tr, va, seed, 4000)
    row = {"parameters": info["parameters"], "train_seconds": info["train_seconds"], "best_validation_auc": info["best_validation_auc"]}
    for name, data in (("same", same), ("near_width", near), ("far_width", far), ("renamed", ren)):
        m = evaluate(model, data); r = rec_seed["models"][kind][name]
        row[name] = {"auc": m["auc"], "recorded_auc": r["auc"], "abs_diff": abs(m["auc"] - r["auc"])}
    if kind == "dag":
        with torch.no_grad():
            row["max_relabel_logit_diff_all_512"] = max(float(torch.max(torch.abs(model([a])[0] - model([b])[0]))) for a, b in zip(same, ren))
    report["models"][kind] = row
    (out / "spot_reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(kind, json.dumps(row), flush=True)
