# Classifier-free guidance follow-up

The repaired model was trained with ten percent null-target dropout, but its
unguided validation result remained dominated by the unconditional circuit
prior. Before running this follow-up, freeze classifier-free guidance scales
1, 2, 4, and 8. Use the existing 64-target validation split and matched
best-of-8 streams. Select the scale with maximum mean correct-condition
fidelity, then evaluate the already generated and untouched 64-target final
split exactly once at that scale. Report correct, cyclically permuted, and null
conditions and the paired bootstrap interval.

Guidance applies `null + scale * (conditional - null)` to both action logits
and bounded angle residuals. Scale 1 reproduces ordinary conditional sampling.
This is a validation-selected guided-generation result, not an unguided model
or representation-superiority claim.
