import json, time, sys, warnings
warnings.simplefilter("ignore")
import numpy as np
from collections import defaultdict
from qdag.algorithm_suite import FAMILIES, generate_suite, commutation_labels
from qdag.schedule_benchmark import examples_hash
out = {}
SPLITS = {"train_like": (90100, 220, (4, 7)), "near_like": (90103, 110, (8, 12)), "far_like": (90104, 110, (13, 20))}
for name, (seed, count, qubits) in SPLITS.items():
    t = time.time(); ex, fam, comp = generate_suite(seed, count, qubits, return_compiled=True); gen = time.time() - t
    ex2, _ = generate_suite(seed, 22, qubits)
    reproducible = examples_hash(ex[:22]) == examples_hash(ex2)
    per = defaultdict(list)
    t = time.time(); comm = [commutation_labels(c) for c in comp]; tc = time.time() - t
    for e, f, c in zip(ex, fam, comm):
        gates = [o.gate for o in e.ops]
        per[f].append((e.n_qubits, len(e.ops), float(e.labels.mean()), float(c.mean()), float((c == e.labels).mean()), gates.count("h") / len(gates)))
    rows = {}
    for f, v in per.items():
        a = np.asarray(v)
        rows[f] = {"n": len(v), "gates_min": int(a[:, 1].min()), "gates_mean": round(float(a[:, 1].mean()), 1), "gates_max": int(a[:, 1].max()),
                   "pos_wire": round(float(a[:, 2].mean()), 3), "pos_comm": round(float(a[:, 3].mean()), 3), "label_agree": round(float(a[:, 4].mean()), 3)}
    allv = np.concatenate([np.asarray(v) for v in per.values()])
    nodes = int(allv[:, 1].sum())
    out[name] = {"gen_seconds": round(gen, 1), "comm_label_seconds": round(tc, 1), "reproducible_prefix": reproducible, "circuits": len(ex), "nodes": nodes,
                 "gates_mean": round(nodes / len(ex), 1), "gates_max": int(allv[:, 1].max()),
                 "node_pos_wire": round(float(np.concatenate([e.labels for e in ex]).mean()), 3), "node_pos_comm": round(float(np.concatenate(comm).mean()), 3),
                 "node_label_agree": round(float(np.mean(np.concatenate(comm) == np.concatenate([e.labels for e in ex]))), 3),
                 "unique": len({examples_hash([e]) for e in ex}), "families": rows}
    print(name, json.dumps({k: v for k, v in out[name].items() if k != "families"}), flush=True)
    for f, r in rows.items(): print("   ", f, r, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2)
