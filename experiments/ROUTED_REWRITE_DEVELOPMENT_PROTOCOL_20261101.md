# Frozen development protocol: routed natural-template rewrite ranking

## Purpose

Test whether a wire-dependency DAG predicts the downstream hardware-routing value of safe compiler rewrites better than a matched canonical-grid encoder. This replaces source critical-path prediction with an amortized compiler-oracle task: exhaustive evaluation must recompile every candidate for a constrained coupling map, while a learned ranker receives one compiler call.

## Data

Circuits come from four deterministic parametric families: QAOA-like alternating operators, hardware-efficient VQE, reversible linear arithmetic and chemistry-inspired excitation blocks. Safe candidates arise from three compiler patterns: parameter-bound zero rotations, inverse rotation cancellation and self-inverse CNOT cancellation. Independent gates may occur between inverse operations, modelling templates exposed after commutation or composition of separately generated blocks.

Development training uses 4–7 qubits and 32–64 base operations from QAOA, VQE and arithmetic. Validation uses fresh circuits from the same range and families. Development evaluation uses 8–10 qubits and 72–104 operations from the same families plus held-out chemistry circuits. Every candidate is checked by statevector fidelity on tractable development instances.

## Oracle and endpoint

Compile the original and every rewritten circuit to a bidirectional line coupling map with Qiskit, fixed seed 17, basis `rz,sx,x,cx`, optimization level 0 and a fixed trivial initial layout. Candidate utility is the reduction in routed CX count, with routed depth reduction as a secondary metric. The primary per-circuit target is the candidate with greatest CX reduction, breaking ties by depth reduction.

The learned method selects one candidate and receives one downstream compilation. Report fraction of oracle CX benefit recovered, mean CX and depth reduction, exact oracle-choice rate and regret. Compare DAG, canonical grid, exact source critical-path scoring, source-depth lookahead, first-candidate production order and random choice.

## Development gate

Run one paired seed with identical training examples, optimizer, updates and candidate targets. Continue to a frozen final test only if DAG exceeds grid on recovered routed-CX benefit and beats or matches the strongest cheap heuristic. Development results cannot support the paper's final claim. If routing at optimization level 0 creates no meaningful candidate variation, revise the oracle before any final protocol and record this development result.
