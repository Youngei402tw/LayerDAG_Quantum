# Supplemental gate-set repair protocol

Declared after the initial pilot: its x-forbidden branch generated zero cases because no level-0 circuit contained x. This is a coverage correction, not an independent held-out discovery experiment.

Reuse all 24 frozen pilot targets and seeds. Prohibit sx explicitly and allow ry/rz/x/cx. Replace every sx in both the level-0 and level-3 circuits by rz(pi/2), ry(pi/2), rz(-pi/2) in execution order. This equals sx up to global phase and repairs the gate mask but increases gate count. No compression claim. Compare original seed, repaired seed, original optimized compiler, repaired optimized compiler. Independent NumPy and Qiskit verification of every result; record fidelity, resources, origin and wall time. No training or parameter optimization.
