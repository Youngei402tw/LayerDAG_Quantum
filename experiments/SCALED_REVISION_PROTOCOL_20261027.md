# Scaled repaired-model protocol

The 256-target repaired run fit its monitored training actions but generalized
poorly, and classifier-free guidance could not recover target dependence. This
follow-up tests the pre-result diagnosis that a 117,848-parameter model was
memorizing too few continuously parameterized circuits.

Freeze 2,048 fresh training targets (seed 20261030), 128 validation targets
(seed 20261031), and 128 final targets (seed 20261101), with phase-invariant
duplicate rejection within and across splits. Use the unchanged width-64
factorized masked DAG model, optimizer, target dropout, batch construction, and
ordinary conditional sampling. Train seed 30 for at most 30,000 updates or 15
minutes. To keep evaluation bounded, select checkpoints on a deterministic
2,048-example subset of the training prefixes every 250 updates.

Evaluate validation with matched best-of-8 correct, cyclically permuted, and
null conditions. Evaluate the final split only if the validation bootstrap
interval for paired correct-minus-permuted fidelity lies strictly above zero.
No guidance-scale or architecture tuning follows this validation result.
