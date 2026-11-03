# What this paper contributes

## The problem

Quantum-circuit models are usually trained on small circuits because generating labels or simulating larger circuits is expensive. A model can perform well on other small circuits by learning regular positions in a circuit layout. Those positions change when the circuit becomes wider and longer.

## The experiment

The paper compares two nearly equal-size models with the same gate information and training budget:

- **Layer grid:** represents a gate by its canonical position among layers and wires.
- **Wire DAG:** connects each gate to the previous and next operations on the qubit wires it uses.

The task asks which gates lie on a maximum dependency path. This is deliberately an exact diagnostic task. It measures whether the representation supports dependency reasoning without introducing simulator noise or uncertain labels. The neural model is not proposed as a replacement for the exact critical-path algorithm.

## The finding

The best representation changes with the evaluation regime:

| Evaluation | DAG AUROC | Grid AUROC | Winner |
|---|---:|---:|---|
| Familiar circuit sizes | 0.8556 | **0.9406** | Grid |
| Wider and longer circuits | **0.8078** | 0.7861 | DAG |

The DAG wins the larger-circuit comparison in all three paired training seeds. Therefore, validation on familiar sizes would select the grid, even though the DAG transfers better to larger circuits.

A second frozen study tests structured QAOA-like, VQE and arithmetic circuits with five new paired seeds. Under simultaneous width-and-depth shift, DAG reaches 0.8902 AUROC versus 0.8495 for grid and wins every seed. Separating the axes shows +0.0514 AUROC from width shift and only +0.0025 from depth shift. The advantage is therefore associated mainly with adding qubit wires. On a held-out chemistry-inspired family, grid AUROC is higher and both models are weak; size transfer does not imply family transfer.

## Why it happens

Grid coordinates such as “wire 2, layer 5” are useful and easy to memorize when circuit dimensions are familiar. Adding wires and layers changes those coordinates. A DAG edge means “the previous operation on this wire,” which retains the same meaning at different widths and lengths. The DAG pays for that structural bias with lower in-distribution accuracy, but it loses less under the tested size shift.

## What is new

The contribution is controlled evidence across two studies and eight paired seeds that **explicit wire dependencies improve within-family extrapolation when quantum circuits gain qubit wires**. The random-circuit study also shows that in-distribution accuracy can select the wrong encoding. Both studies match parameter count, features, data, optimization steps, and paired seeds; freeze final gates in advance; and report negative results.

## What is not claimed

The paper does not claim that DAGs are always better, that critical paths require machine learning, or that the current system improves a production compiler. Three downstream attempts did not establish that practical claim. They are retained to define the boundary of the evidence.

## Best paper positioning

Present this as a controlled representation and generalization study for quantum-circuit learning. The practical motivation is that learned compiler and circuit models often train on smaller instances than deployment workloads. The result supports testing representations under explicit size shift rather than choosing them solely from same-size validation.
