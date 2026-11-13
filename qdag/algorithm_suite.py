"""PROTOTYPE (scoping only, 2026-11-13): randomized standard-algorithm circuit families.

Built with Qiskit's circuit library and compiled by the Qiskit transpiler to the {H, RY, RZ, CX}
basis that the frozen QAI encoders already read, so no model code changes. This module exists to
support the DRAFT protocol experiments/DRAFT_QAI_ALGORITHM_SUITE_PROTOCOL_20261113.md. No model
has been trained or evaluated on its output. Family list and parameters are not frozen.
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from .core import Op
from .schedule_benchmark import CircuitExample, critical_labels, examples_hash

BASIS = ("h", "ry", "rz", "cx")
MIN_GATES = 8


def _lib():
    import qiskit.circuit.library as library
    return library


def _qft(rng, n):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    for q in range(n): qc.ry(float(rng.uniform(0, math.pi)), q)  # random product input state
    qc.append(_lib().QFTGate(n), range(n)); return qc


def _qpe(rng, n):
    from qiskit import QuantumCircuit
    m = n - 1; phase = float(rng.integers(1, 2 ** m)) / 2 ** m if rng.random() < .5 else float(rng.random())
    qc = QuantumCircuit(n); qc.x(m); qc.h(range(m))
    for k in range(m): qc.cp(2 * math.pi * phase * 2 ** k, k, m)
    qc.append(_lib().QFTGate(m).inverse(), range(m)); return qc


def _bv(rng, n):
    from qiskit import QuantumCircuit
    secret = rng.integers(0, 2, n - 1)
    if not secret.any(): secret[int(rng.integers(n - 1))] = 1
    qc = QuantumCircuit(n); qc.x(n - 1); qc.h(range(n))
    for i, bit in enumerate(secret):
        if bit: qc.cx(i, n - 1)
    qc.h(range(n - 1)); return qc


def _random_graph(rng, n):
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < min(1., 3 / n)]
    return edges or [(0, 1)]


def _graphstate(rng, n):
    from qiskit import QuantumCircuit
    adjacency = np.zeros((n, n), int)
    for i, j in _random_graph(rng, n): adjacency[i, j] = adjacency[j, i] = 1
    qc = QuantumCircuit(n); qc.append(_lib().GraphStateGate(adjacency), range(n)); return qc


def _qaoa(rng, n):
    from qiskit.quantum_info import SparsePauliOp
    cost = SparsePauliOp.from_sparse_list([("ZZ", [i, j], 1.0) for i, j in _random_graph(rng, n)], n)
    ansatz = _lib().qaoa_ansatz(cost, reps=int(rng.integers(1, 3)))
    return ansatz.assign_parameters(rng.uniform(-math.pi, math.pi, ansatz.num_parameters))


def _vqe(rng, n):
    entanglement = ("linear", "circular", "pairwise", "reverse_linear")[int(rng.integers(4))]
    build = _lib().real_amplitudes if rng.random() < .5 else _lib().efficient_su2
    ansatz = build(n, entanglement=entanglement, reps=int(rng.integers(1, 4)))
    return ansatz.assign_parameters(rng.uniform(-math.pi, math.pi, ansatz.num_parameters))


def _trotter(rng, n):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.synthesis import LieTrotter
    pairs = [(i, i + 1) for i in range(n - 1)] + ([(n - 1, 0)] if rng.random() < .5 else [])
    terms = []
    if rng.random() < .5:  # Heisenberg chain or ring with random couplings and Z fields
        for i, j in pairs: terms.extend((p, [i, j], float(rng.uniform(.5, 1.5))) for p in ("XX", "YY", "ZZ"))
        terms.extend(("Z", [i], float(rng.uniform(-1, 1))) for i in range(n))
    else:  # transverse-field Ising
        terms.extend(("ZZ", [i, j], float(rng.uniform(.5, 1.5))) for i, j in pairs)
        terms.extend(("X", [i], float(rng.uniform(.5, 1.5))) for i in range(n))
    hamiltonian = SparsePauliOp.from_sparse_list(terms, n); qc = QuantumCircuit(n)
    synthesis = LieTrotter(reps=int(rng.integers(1, 4)))
    qc.append(_lib().PauliEvolutionGate(hamiltonian, time=float(rng.uniform(.1, .5)), synthesis=synthesis), range(n)); return qc


def _adder(rng, n):
    from qiskit import QuantumCircuit
    if n % 2: adder = _lib().CDKMRippleCarryAdder((n - 1) // 2, kind="fixed")
    else: adder = _lib().CDKMRippleCarryAdder((n - 2) // 2, kind=("full", "half")[int(rng.integers(2))])
    qc = QuantumCircuit(adder.num_qubits)
    for q in range(adder.num_qubits):  # random classical inputs
        if rng.random() < .5: qc.x(q)
    qc.compose(adder, inplace=True); return qc


def _qv(rng, n):
    return _lib().quantum_volume(n, depth=int(rng.integers(1, 4)), seed=int(rng.integers(1 << 30)))


def _iqp(rng, n):
    weights = np.zeros((n, n), int)
    for i, j in _random_graph(rng, n): weights[i, j] = weights[j, i] = int(rng.integers(1, 8))
    weights += np.diag(rng.integers(0, 8, n)); return _lib().iqp(weights)


def _excitation(rng, n):
    from qiskit import QuantumCircuit
    ansatz = _lib().excitation_preserving(n, entanglement=("linear", "circular")[int(rng.integers(2))], reps=int(rng.integers(1, 3)))
    qc = QuantumCircuit(n)
    for q in range(n):  # random occupation pattern
        if rng.random() < .5: qc.x(q)
    return qc.compose(ansatz.assign_parameters(rng.uniform(-math.pi, math.pi, ansatz.num_parameters)))


FAMILIES = {"qft": _qft, "qpe": _qpe, "bv": _bv, "graphstate": _graphstate, "qaoa": _qaoa, "vqe": _vqe,
            "trotter": _trotter, "adder": _adder, "qv": _qv, "iqp": _iqp, "excitation": _excitation}


def compile_circuit(qc):
    """Qiskit transpiler, optimization level 1, no coupling map: basis translation plus light cleanup."""
    from qiskit import transpile
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return transpile(qc, basis_gates=list(BASIS), optimization_level=1, seed_transpiler=0)


def to_ops(compiled) -> list[Op]:
    ops = []
    for instruction in compiled.data:
        name = instruction.operation.name
        if name not in BASIS: raise ValueError(f"unexpected gate {name}")
        wires = tuple(compiled.find_bit(q).index for q in instruction.qubits)
        ops.append(Op(name, wires, tuple(float(p) for p in instruction.operation.params)))
    return ops


def generate_suite(seed: int, count: int, qubits: tuple[int, int], families: tuple[str, ...] = tuple(FAMILIES),
                   exclude: set[str] | None = None, return_compiled: bool = False):
    """Returns (examples, family_names[, compiled circuits]). Rejects circuits below MIN_GATES and exact duplicates."""
    rng = np.random.default_rng(seed); examples = []; names = []; compiled_list = []
    seen = set(exclude or ())
    for index in range(count):
        family = families[index % len(families)]
        for attempt in range(200):
            n = int(rng.integers(qubits[0], qubits[1] + 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                compiled = compile_circuit(FAMILIES[family](rng, n))
            ops = to_ops(compiled)
            if len(ops) < MIN_GATES: continue
            example = CircuitExample(compiled.num_qubits, ops, critical_labels(compiled.num_qubits, ops))
            key = examples_hash([example])
            if key in seen and attempt < 199: continue
            seen.add(key); examples.append(example); names.append(family); compiled_list.append(compiled); break
        else:
            raise RuntimeError(f"could not generate {family} for qubits {qubits}")
    return (examples, names, compiled_list) if return_compiled else (examples, names)


def commutation_labels(compiled) -> np.ndarray:
    """Membership in a longest chain of Qiskit's commutation-aware dependency DAG (DAGDependency)."""
    from qiskit.converters import circuit_to_dagdependency
    dag = circuit_to_dagdependency(compiled); count = dag.size()
    before = np.ones(count, np.int64); after = np.ones(count, np.int64)
    for node in range(count):
        parents = dag.direct_predecessors(node)
        if parents: before[node] = 1 + max(before[p] for p in parents)
    for node in range(count - 1, -1, -1):
        children = dag.direct_successors(node)
        if children: after[node] = 1 + max(after[c] for c in children)
    return (before + after - 1 == before.max()).astype(np.float32)
