# Data only, scoping seeds: can every family supply the unique circuits the final split sizes need?
import warnings, time, json, sys; warnings.simplefilter("ignore")
import numpy as np
from collections import defaultdict
from qdag.algorithm_suite import generate_suite, commutation_labels
from qdag.schedule_benchmark import examples_hash
t = time.time(); seen = set(); out = {}
for name, seed, count, qubits in (("training", 90200, 1100, (4, 7)), ("validation", 90201, 264, (4, 7)), ("same", 90202, 528, (4, 7)),
                                  ("near", 90203, 528, (8, 12)), ("far", 90204, 528, (13, 20))):
    ex, fam, comp = generate_suite(seed, count, qubits, exclude=seen, return_compiled=True)
    keys = [examples_hash([x]) for x in ex]; assert len(set(keys)) == len(keys) and not seen & set(keys); seen |= set(keys)
    t1 = time.time(); comm = [commutation_labels(c) for c in comp]; tc = time.time() - t1
    per = defaultdict(lambda: [0, 0, 0.0, 0.0, 0])
    for x, f, c in zip(ex, fam, comm):
        r = per[f]; r[0] += 1; r[1] += len(x.ops); r[2] += float(x.labels.sum()); r[3] += float(c.sum()); r[4] = max(r[4], len(x.ops))
    out[name] = {"circuits": len(ex), "nodes": sum(len(x.ops) for x in ex), "comm_label_seconds": round(tc, 1),
                 "families": {f: {"n": r[0], "gates_mean": round(r[1] / r[0], 1), "gates_max": r[4], "pos_wire": round(r[2] / r[1], 3), "pos_comm": round(r[3] / r[1], 3)} for f, r in per.items()}}
    print(name, {k: v for k, v in out[name].items() if k != "families"}, flush=True)
print("oracle:", {k: v["families"]["oracle"] for k, v in out.items()})
print("total seconds", round(time.time() - t, 1)); json.dump(out, open(sys.argv[1], "w"), indent=2)
