# TANGO v3: Counterfactual Latent-Memory Credit Protocol

## Research question

TANGO v3 tests whether downstream GUI return can be assigned to the historical
visual KV block that causally carries useful evidence. It does **not** treat a
higher offline candidate margin as evidence of a better executable policy.

For decision state `h_t`, action `a`, and historical block `m_j`, estimate two
separate quantities:

```text
Action credit:  A_action(h_t,a) = Q(h_t,a) - V(h_t)
Memory credit:  A_memory(t,j)   = Q(h_t,M_t) - Q(h_t,M_t^{-j})
```

Action credit trains execution. Memory credit trains only a state-conditioned
K/V gate and transport. Episode return broadcast is retained as a baseline.

Three interventions are reported separately: (i) removing a screenshot and
re-encoding tests dependence on historical visual evidence; (ii) zeroing a
decoder K/V span tests a latent path but may be distribution-shifting; (iii)
the primary KV diagnostic patches the target span with a matched distractor
span from the same encoded sequence, preserving token count and activation
scale. Only (ii)/(iii) are called KV-block interventions.

## Claims and falsification criteria

1. **Causal memory sensitivity.** Removing a cue block must hurt future return
   more than removing a matched distractor block.
2. **Credit localization.** Memory advantage must identify useful, stale, and
   irrelevant history more accurately than global-return or action advantage.
3. **Executable improvement.** The learned transport must improve critical-fork
   success, not only teacher-forced margin.
4. **Low-energy invariance.** Improvements must not substantially perturb
   non-critical states.
5. **Generalization.** Gains must survive unseen compositions and render
   templates before real-benchmark or no-hook distillation claims are made.

A claim fails when its pre-registered gate below fails. Failed variants remain
reported rather than being selected by test performance.

## Controlled visual tasks

The existing `hidden_memory` and `distractor_credit` families remain mechanism
sanity checks. Two added probes are:

- `multi_cue_binding`: color and symbol occur in different history blocks; both
  are needed at a delayed fork.
- `interference_update`: a newer cue supersedes an obsolete cue; useful memory
  must be retained while stale memory is suppressed.

Their semantic pilot is intentionally a difficulty preflight, not a training
benchmark. If base critical accuracy is outside 35–85%, use the hard variants:

- **Nonce visual binding:** replace semantic color/symbol names with randomized
  glyph IDs and render-only patterns; action labels are neutral slot IDs.
- **Compositional OOD:** train on seen glyph/color marginals and test unseen
  pairings, layouts, font scales, and option permutations.
- **Interference chains:** show 2–4 successive updates with random delays; only
  the latest valid update determines return.
- **Matched distractors:** visually match cue pages while carrying no task
  information, preventing layout-based localization.

Every generated instance records the ground-truth critical block label in a
side channel never included in the policy prompt.

`useful/stale/irrelevant` is a designer-provided semantic role, not an assumed
model-causal sign. The empirical sign is always defined by the measured policy
or rollout difference. A stale block may still help a frozen model interpret
an update relation; such a result falsifies the proposed negative label rather
than being overwritten by it.

## Splits

| Split | Seeds | Templates | Purpose |
|---|---:|---|---|
| train | 0–199 | A/B | optimize modules |
| validation | 200–249 | A/B, new compositions | select fixed hyperparameters |
| ID test | 250–449 | A/B | paired estimate |
| template OOD | 450–649 | C/D only | rendering generalization |
| family holdout | separate run | one family excluded | task-rule transfer |

No test split is used for checkpoint or layer selection. A 20-seed pilot is
used only for difficulty gating; accepted task templates are frozen afterward.

## Fair method matrix

All learned methods use the same backbone, candidate states, history images,
train/validation split, number of optimizer steps, and parameter budget.

| ID | Method | Training signal |
|---|---|---|
| M0 | Base/Warm | no adaptation |
| M1 | Critical Action CE | oracle current action only |
| M2 | Successful-only CE | actions from successful trajectories |
| M3 | Global Return-KV | final return broadcast to every step |
| M4 | Action Advantage | same-prefix `Q(h,a)-V(h)` |
| M5 | Fork-DPO | pairwise successful/failed fork preference |
| M6 | Shuffled Memory Advantage | memory-credit sanity control |
| M7 | Oracle Memory Gate | ground-truth causal blocks; upper bound |
| M8 | TANGO-Memory | estimated `Q(M)-Q(M^{-j})` |
| M9 | TANGO-Memory+Energy | M8 plus state-conditioned gate and energy |
| M10 | TANGO Two-stage | gate localization, then frozen-gate transport |

M1–M6 receive matched examples and optimizer updates. M7 is not a comparable
learning result and is labeled as an oracle ceiling. Transport rank and layer
count are matched across KV methods. On-policy branching is used for the final
claim; immutable offline continuations are reported as a separate ablation.

## Diagnostics before training

1. Critical-only CE: establishes representation/action learnability.
2. History oracle: repeats the relevant evidence at the decision point.
3. Cue-versus-distractor KV ablation: measures causal probability/return drop.
4. K/V/layer scan: selects a layer set using validation only.

If history oracle fails, stop KV training and repair the prompt/visual task. If
cue ablation is indistinguishable from distractor ablation, stop and repair the
memory path. Do not choose `V-only, last-8` by convention.

The K/V scan reports correct-action score effect
`log p(a*|M)-log p(a*|patch_j(M))`, margin effect, critical accuracy, and sign
agreement. Matched patching is primary; zero ablation is a stress control.

## Metrics and statistical protocol

Primary metrics:

- executable critical-fork accuracy and full-trajectory success;
- paired TANGO-minus-baseline success difference;
- memory localization AUROC/AUPRC using `|A_memory(t,j)|` or gate score;
- signed stale-memory accuracy for update tasks;
- non-critical policy perturbation (mean absolute log-probability change);
- transport energy and improvement per unit energy;
- per-family win/loss/tie and illegal-action rate.

Report task-family clustered bootstrap 95% confidence intervals and paired
McNemar tests. Seeds share a task template and are not treated as independent
task families.

## Pre-registered gates

- Difficulty: base critical accuracy is 35–85% on the frozen pilot.
- Causality: cue-removal effect exceeds distractor-removal effect with 95% CI.
- Localization: test AUROC >= 0.80 and correct stale-memory sign >= 80%.
- Method: TANGO exceeds both Action CE and Action Advantage by >= 5 percentage
  points on critical success, with paired 95% CI excluding zero.
- Invariance: mean non-critical absolute perturbation <= 0.05 and illegal
  actions do not significantly increase.
- Generalization: OOD gain retains at least 70% of the ID-test gain.
- Distillation: run only after the teacher passes the method gate; residual
  no-hook student must preserve >= 70% of the teacher's online gain.

## Execution order

1. Freeze hard task generator after the 20-seed difficulty preflight.
2. Run four diagnostics and validation-only K/V/layer selection.
3. Run M0–M10 on 200 train seeds and one fixed optimization seed.
4. Repeat the three strongest non-oracle methods over three training seeds.
5. Evaluate ID, template-OOD, and family-holdout tests with paired statistics.
6. Only after passing the controlled gate, run MemGUI-Bench and a stable
   AndroidWorld subset using on-policy action and memory interventions.
7. Only after an online teacher win, train residual trajectory-policy
   distillation; do not distill a margin-only teacher.

## Current pilot status (2026-08-27)

The first five-seed semantic pilot has 80 prefixes and 10 critical forks.
Qwen2.5-VL-3B scores 10/10 with full history, so it fails the difficulty gate
and is retained only as a causal probe. Dropping history block 0 gives 9/10,
dropping block 1 gives 6/10, and dropping the distractor block gives 10/10.
Thus the visual history path is causal, but harder nonce/OOD variants are
required before comparing learning methods.

A second ten-seed `nonce_visual_binding` pilot used render-only random codes,
neutral `select_slot_i` actions, six choices, and 10 critical forks. Base Qwen
again scored 10/10, so semantic action leakage was not the cause of saturation.
This variant is also rejected for comparative training. The next difficulty
preflight must combine 2–4 cue updates, matched visual distractors, and unseen
compositions/templates; no M0–M10 sweep is authorized until the 35–85% gate is
met.

The ten-seed `interference_chain` pilot then introduced five historical
records (initial, matched reference, update, matched reference, latest update)
and eight neutral slots. Base Qwen scored 6/10 critical forks, passing the
35–85% difficulty gate. It is the first accepted training-comparison family;
the next required gate is matched KV-patch localization on its five blocks.

That gate passed on the 10-seed pilot at middle-layer K. Replacing the latest
update with a matched reference reduced critical accuracy from 60% to 0% and
gave mean correct-action score effect `+0.237`. Replacing the superseded update
raised accuracy from 60% to 100% with effect `-0.096`. Two references left
accuracy at 60% with effects `-0.019` and `+0.001`. Across the two updates and
two references, absolute-effect localization AUROC is 1.0, update sign accuracy
is 1.0, and both references fall within the predeclared neutral band `|effect|
<= 0.025`. The initial record has weak positive effect `+0.047`, demonstrating
why semantic role labels must not overwrite empirical causal signs.
