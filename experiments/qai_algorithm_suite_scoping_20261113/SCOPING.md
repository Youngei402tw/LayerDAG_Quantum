# Scoping note: a realistic-circuit experiment for the QAI paper

Written 2026-11-13. **Nothing here is frozen and no model has been trained or evaluated on any circuit described here.** The only model code that ran is `time_untrained_models.py`, which times untrained networks and computes no accuracy. Scoping used throw-away data seeds 90100–90104; they must never be used for a final run.

## What the paper still lacks

After the connectivity and augmentation controls, the remaining reviewer objections are (1) every circuit is synthetic, from three hand-written generators, and (2) the probe label (membership in a longest wire-dependency path) is computable in linear time and ignores gate semantics, which is also why operand roles turned out not to matter.

## Options considered

| Option | Addresses | Feasible here | Main risk | Verdict |
|---|---|---|---|---|
| **A. Standard-algorithm suite, same probe.** 11 families from Qiskit's circuit library with randomized instances, compiled by the Qiskit transpiler to {H, RY, RZ, CX}; train 4–7 qubits, test 13–20; the four key encoders unchanged | (1) | Yes; measured below | Result may differ from the synthetic study; must be reported either way | **Recommended (primary)** |
| **B. Same circuits, commutation-aware label.** Membership in a longest chain of Qiskit's commutation-aware dependency DAG (`DAGDependency`): RZ commutes through a CX control but not its target, CX gates sharing only a control or only a target commute, and so on | (2), and gives operand roles a job | Yes; labels for 528 far-width circuits take under a minute | Six rounds of wire-level message passing may be too weak; no tuning allowed | **Recommended (secondary)** |
| C. Duration-weighted slack (virtual RZ, long CX) | part of (2) | Yes | Still linear-time and less quantum-specific than B | Not recommended over B |
| D. Noisy-fidelity regression (QuEst-style) | (2) with a genuinely expensive label | Needs a new simulator dependency, circuit-level heads (new architectures), and noisy simulation at 13–20 qubits | Under uniform noise a gate-count baseline is likely to saturate, as the smallest-angle heuristic did in the deletion study | Not recommended now |
| E. Routing-overhead prediction | (2), compiler-facing | Labels from stochastic routing are not invariant to relabeling, so the symmetry argument no longer applies cleanly | The two earlier routed studies in this repository did not favor the DAG | Not recommended |
| F. Public MQT Bench instances as an evaluation-only external set | (1), strongest form | **Blocked in this session**: installing `mqt.bench` was refused as new external code. A dry run also showed it would upgrade the pinned Qiskit 2.2.3 to 2.6.2, so it needs an isolated environment with circuits exported as OpenQASM | Owner must approve or run the install | Optional add-on to A |

Neither A nor B shows practical utility: both labels remain exactly computable. A tests whether the paper's conclusions hold on real algorithm structure; B is a harder, quantum-specific probe.

## Feasibility measurements (prototype `qdag/algorithm_suite.py`)

Families: QFT on a random product state, phase estimation of a random phase, Bernstein–Vazirani, random graph states, MaxCut QAOA on random graphs, hardware-efficient VQE ansätze (RealAmplitudes, EfficientSU2), Trotterized Heisenberg and transverse-field Ising evolution, CDKM ripple-carry adders on random inputs, quantum-volume layers, random IQP circuits, and the excitation-preserving ansatz. All 11 compile to the four-gate basis at 4–20 qubits; all four gate types occur. Generation is deterministic given the seed and takes about a second per hundred circuits.

| Scoping split | Circuits | Gates mean / max | Positive rate, wire label | Positive rate, commutation label | Node agreement between labels |
|---|---|---|---|---|---|
| 4–7 qubits | 220 | 64 / 462 | 0.767 | 0.571 | 0.714 |
| 8–12 qubits | 110 | 139 / 792 | 0.767 | 0.534 | 0.688 |
| 13–20 qubits | 110 | 282 / 1,320 | 0.785 | 0.511 | 0.658 |

Per-family figures are in `data_statistics.json`. Points that shape the protocol:

- Circuit sizes differ by more than 20× between families at far width (Bernstein–Vazirani about 24 gates, QFT about 700), so a node-pooled metric would be dominated by a few families. The draft uses a family-macro AUROC as primary and keeps the pooled AUROC for continuity.
- The commutation label differs from the wire label on about a third of nodes and by far more in some families (QFT agreement 0.15 at far width). Bernstein–Vazirani is all-positive under the commutation label, so it has no within-family AUROC there.
- Circuits exceed the Transformers' 192 learned positions, so the Transformers are excluded.

Measured cost on 2 threads (untrained models): role DAG 0.113 s per update, grid GRU 0.208, pointer GRU 0.212, augmented pointer GRU 0.217. Estimated per seed including evaluation: 471, 863, 881 and 899 s. One phase (4 models × 5 seeds) is about 4.3 CPU-hours sequentially, about 2.2 hours as two parallel processes.

## Decisions for the owner

1. Run A only, or A and B (about twice the compute)?
2. In B, also train the role-ablated DAG (about 40 more minutes), to test whether operand roles matter once the label depends on control versus target?
3. Allow an isolated MQT Bench install for an evaluation-only public set (option F)?
4. Where do the results go? The manuscript is at the 6-page limit. Either compress the earlier supporting studies, use paid extra pages where a venue allows them, or target a venue with a longer limit.
