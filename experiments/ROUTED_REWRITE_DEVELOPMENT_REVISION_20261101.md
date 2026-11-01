# Development-only routed rewrite revision

The first development run is retained and its gate invalidated. `cx_recovery` clipped negative selected CX reductions while `mean_cx_reduction` did not, allowing the nominal gate to pass even though DAG mean CX reduction (6.8047) was slightly below grid (6.8281). No final data were generated or inspected.

Revision 1 changes recovery to selected total CX reduction divided by oracle total CX reduction, preserving harmful choices. The primary gate now directly requires DAG mean CX reduction above grid. The hard single-best cross-entropy target is replaced with a listwise target proportional to `softmax((CX reduction + 0.05 depth reduction)/2)`, so near-equivalent candidates receive credit and harmful candidates receive less probability. Validation checkpointing uses mean CX reduction. All other data, architectures, features, optimizer settings and controls remain fixed.
