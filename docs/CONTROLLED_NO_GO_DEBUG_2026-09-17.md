# Controlled synthetic NO-GO debug, 2026-09-17

This note diagnoses the controlled interference-chain run recorded in
`docs/EXPERIMENT_STATUS_2026-09-17.md`. The opened template-C test result was
used only for post-hoc failure analysis, not for selecting new hyperparameters.

## Symptom

Validation selected the memory arm, but the sealed test did not improve
discrete candidate accuracy:

| Split | Arm | Accuracy | Mean action NLL |
|---|---|---:|---:|
| Validation | Base | 0.16 | 0.402201 |
| Validation | Memory, alpha 20 | 0.18 | 0.402121 |
| Test | Base | 0.120 | 0.412617 |
| Test | Memory, alpha 20 | 0.115 | 0.412004 |

The memory arm slightly improved NLL but regressed discrete test accuracy by
one decision.

## Paired decision audit

On validation, memory changed only 4/50 decisions. It rescued 1 example and
regressed 0. The apparent validation gain is therefore a one-example effect.

On the already-opened test, memory changed 16/200 decisions. It rescued 2
examples, regressed 3, and left 184 unchanged.

The validation target distribution was also favorable to the model's native
slot bias: 13/50 validation examples had target index 6, and the base model
already selected index 6 in 29/50 examples. Test targets were more evenly
distributed.

## Score movement

The intervention was very small:

- validation mean true-action score delta: `0.000080`
- test mean true-action score delta: `0.000613`
- validation mean top-score delta: `0.000519`
- test mean top-score delta: `-0.001472`

The bank weights also remained tiny after training; representative
`up.weight` means were around `0.001`. The method mostly nudged existing
candidate priors instead of creating a robust decision change.

## Credit-controller failure

The train teacher credits have real signed variation:

| Block | Target mean credit | Target std | Positive / Negative / Neutral |
|---:|---:|---:|---|
| 0 | 0.001696 | 0.037355 | 83 / 106 / 11 |
| 1 | -0.001589 | 0.026324 | 93 / 97 / 10 |
| 2 | -0.012976 | 0.062385 | 87 / 107 / 6 |
| 3 | 0.009630 | 0.040720 | 110 / 82 / 8 |
| 4 | 0.006474 | 0.029328 | 112 / 73 / 15 |

But the trained controller outputs nearly fixed positive values:

| Eval source | Mean predicted credits by block |
|---|---|
| Validation | `[0.0321, 0.0211, 0.0111, 0.1595, 0.1403]` |
| Test | `[0.0312, 0.0231, 0.0133, 0.1455, 0.1385]` |
| Train diagnostic | `[0.0341, 0.0200, 0.0101, 0.1549, 0.1355]` |

On the 200 training prefixes, the controller-versus-teacher diagnostic was:

- flat credit correlation: `0.1624`
- sign accuracy: `0.512`
- argmax-block accuracy: `0.34`

So the controller did not fit the signed teacher credits even on train. It
learned a weak block-position prior, mainly "blocks 3 and 4 are positive",
instead of learning prefix-specific signed credit.

## Root cause

The NO-GO is primarily a credit-learning failure.

The local-action-logprob teacher produces target-action-conditioned signed
credits, but the deployed controller emits one candidate-independent gate vector
from pooled history features and the query. In this controlled task, the useful
memory depends on matching a historical code to the current slot layout. The
current controller/bank setup does not recover that matching signal; it mostly
learns position bias and produces a very small residual update.

This explains the observed pattern:

- small validation gain from one rescued example;
- slightly better NLL because the target score is nudged on average;
- no robust accuracy gain because top-1 decisions remain dominated by slot
  priors;
- template-C regression when the validation slot-bias coincidence disappears.

## Debugging boundaries

The opened template-C test must remain sealed for future method selection. New
tuning should use train plus validation only, or a newly generated independent
held-out split such as template-D with fresh seeds.

## Next fixes to try

1. Replace local-action-logprob labels with a candidate-value or action-advantage
   teacher for this task, because the current local teacher does not directly
   train candidate separation.
2. Add a validation-only credit diagnostic gate before policy training:
   continue only if train credit correlation and sign accuracy exceed a
   documented threshold.
3. Make the controller candidate-aware or slot-aware without using gold labels,
   for example by conditioning gates on each candidate string/current slot
   representation during scoring.
4. Increase controller capacity only after the credit diagnostic improves; more
   policy epochs alone should not be expected to fix the current failure.
5. Use a fresh template-D held-out split for any new final claim after tuning on
   train/validation.
