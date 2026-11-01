# Final development revision: candidate-aware readout

The corrected listwise development run failed: DAG mean routed-CX reduction was 5.6797 versus 6.0078 for grid. Both learned methods beat source criticality, and DAG slightly exceeded source-depth lookahead, but the representation gate did not pass.

The node-wise head used by both runs assigns a score to each gate and averages gate scores for multi-gate candidates. It cannot explicitly represent that two nodes jointly constitute one inverse or cancellation rewrite. Before any final data exist, revision 2 replaces this with a shared candidate-aware readout. Each encoder returns contextual node states. The identical readout receives the mean and maximum state in the candidate, the circuit mean state, candidate-template one-hot, normalized source position and window size. DAG and grid use width 64; the grid uses a bidirectional two-layer GRU and the DAG uses six bidirectional message-passing layers. Training retains the corrected listwise objective, same data, optimizer and 1,500 updates.

This is the last development revision. Advance only if DAG mean routed-CX reduction exceeds grid and DAG recovery matches or exceeds every cheap heuristic. Otherwise freeze the routed benchmark as negative evidence and make no final claim from it.
