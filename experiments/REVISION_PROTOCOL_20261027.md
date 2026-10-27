# Masked factorized DAG repair protocol

This is a new, user-authorized research iteration after the original bounded
diagnosis.  It repairs three implementation problems found before inspecting
any new split results:

1. checkpoint selection exposed the clean action as a model input;
2. generation began from an absorbing mask, but 44% of training examples
   exposed that clean action;
3. a 1,091-class joint output did not share evidence across per-wire rotation
   and angle choices, while mean graph pooling discarded the active wire
   frontier.

The repaired model always predicts the clean frontier from a mask, factorizes
its logits into action type and per-wire gate/angle tokens, and pools the last
node on each wire alongside the global DAG mean. Ten percent target dropout is
used only to make the null-condition control in-distribution. Half of each
training batch uses the empty prefix so target use cannot be bypassed through
the circuit history.

Freeze before training: 256 training targets (seed 20261027), 64 validation
targets (seed 20261028), and 64 final targets (seed 20261029), with the existing
phase-invariant cross-split duplicate rejection. Train one width-64 DAG model,
seed 27, for at most 10,000 updates or 15 minutes. Evaluate paired correct,
cyclically permuted, and null conditions with matched RNG streams and best-of-8
sampling. Validation passes only if the 10,000-bootstrap 95% interval for
correct-minus-permuted fidelity is above zero. Evaluate the final split only
after that pass. No hyperparameter change follows validation inspection.

This run diagnoses the repaired implementation. It does not retrospectively
alter the original frozen experiment or establish superiority over QFusion.
