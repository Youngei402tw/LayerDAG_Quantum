# Timing only: untrained models, no accuracy is computed or printed.
import time, warnings, json, sys
warnings.simplefilter("ignore")
import numpy as np, torch
from torch.nn import functional as F
from qdag.algorithm_suite import generate_suite
from qdag.invariant_benchmark import InvariantDag, InvariantGrid, relabel
from qdag.connectivity_controls import PointerGrid
torch.set_num_threads(2)
train, _ = generate_suite(90100, 220, (4, 7)); far, _ = generate_suite(90104, 110, (13, 20))
rng = np.random.default_rng(0); out = {}
for name, cls, aug in (("dag", InvariantDag, False), ("grid", InvariantGrid, False), ("pointer_grid", PointerGrid, False), ("aug_pointer_grid", PointerGrid, True)):
    torch.manual_seed(0); model = cls(); opt = torch.optim.AdamW(model.parameters(), lr=3e-4); t = time.monotonic()
    for _ in range(30):
        batch = [train[int(i)] for i in rng.integers(len(train), size=16)]
        if aug: batch = [relabel(x, rng) for x in batch]
        loss = torch.stack([F.binary_cross_entropy_with_logits(y, torch.tensor(x.labels)) for y, x in zip(model(batch), batch)]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    step = (time.monotonic() - t) / 30; model.eval(); t = time.monotonic()
    with torch.no_grad():
        for s in range(0, len(far), 16): model(far[s:s + 16])
    ev_far = (time.monotonic() - t) / len(far); t = time.monotonic()
    with torch.no_grad():
        for s in range(0, len(train), 16): model(train[s:s + 16])
    ev_small = (time.monotonic() - t) / len(train)
    # 4000 updates + 16 validation passes (264 circuits) + test passes: same 528, near 528 (~2.2x small cost), far 528 x3 (far, far-relabeled, far-reversed), same x2 renamings
    est = 4000 * step + 16 * 264 * ev_small + 528 * ev_small * 3 + 528 * ev_small * 2.2 + 528 * ev_far * 3
    out[name] = {"seconds_per_update": round(step, 4), "eval_seconds_per_far_circuit": round(ev_far, 4), "estimated_seconds_per_seed": round(est)}
    print(name, out[name], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=2)
