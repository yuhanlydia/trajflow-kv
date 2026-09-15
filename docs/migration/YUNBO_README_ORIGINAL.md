# TrajFlow-KV

Minimal, reproducible Phase-1 implementation of return-driven low-rank KV
memory transport for VLM GUI agents.

The code freezes a Qwen2.5-VL backbone, injects trainable residual low-rank
projectors into decoder `k_proj` and `v_proj` activations, and optimizes them
with a trajectory-level return-weighted objective plus transport-energy and
orthogonality penalties. It includes a deterministic toy policy for CI and a
Qwen/AITW path for real experiments.

The next controlled experiment is specified in
[`docs/tango_v3_experiment_protocol.md`](docs/tango_v3_experiment_protocol.md).
It separates action credit from latent-memory credit and pre-registers fair
baselines, causal localization, invariance, and OOD gates. The first semantic
multi-cue/update pilot saturated at `10/10` critical forks; removing its first
and second history blocks reduced accuracy to `9/10` and `6/10`, while removing
the distractor stayed `10/10`. It is therefore retained as a causal-memory
probe, not evidence of method improvement. The comparative benchmark will use
hard nonce visual binding and interference-chain variants.

The first `interference_chain` pilot now passes the difficulty gate at `6/10`.
More importantly, matched in-sequence middle-layer K patching localizes signed
credit: latest update `+0.237` (accuracy `6/10 -> 0/10`), superseded update
`-0.096` (`6/10 -> 10/10`), and matched references `-0.019/+0.001` (both stay
`6/10`). This is a latent-memory diagnostic GO, not yet a trained-policy claim.

## Current go/no-go result

The latest controlled run is a **single-task online go / cross-task pending**,
not a broad end-to-end success claim. With Qwen2.5-VL-3B, V-only projectors on the last
eight layers (rank 8, alpha 8), 41 training trajectories and 20 epochs, the
held-out success-minus-failure log-probability margin improved from `0.1965`
to `0.2598`. Shuffled-return training reached `0.1955`, and successful-action
CE reached `0.1874`. However, learned free-form rollouts still achieved no
success on the tested Wi-Fi/Bluetooth system tasks. The exact snapshot is in
[`results/gonogo_current.json`](results/gonogo_current.json).

A follow-up with positive-trajectory-only action CE (`lambda=0.01`) retained
the return margin (`0.2602`) and improved MRR from `0.6250` to `0.6349`, but
Top-1 stayed at `0.5`; its online Wi-Fi rollout emitted legal actions yet
repeated the same swipe for all eight steps and failed. This narrows the next
problem to state-conditioned/fork-point action selection rather than syntax.

State-conditioned fork preferences, click-coordinate hard negatives, and
hierarchical type/parameter ranking subsequently raise held-out return margin
to `0.3685`, action MRR to `0.6722`, and type-conditioned Top-1 to `0.8333`.
On real AndroidWorld Bluetooth, the identical structured policy yields
baseline `0/4` versus Return/Fork/Coordinate-KV `4/4` across one deterministic
and three temperature-0.2 seeds. Wi-Fi still fails because the policy acts
before the quick-settings shade is fully expanded. The next gate is therefore
learned state routing and cross-task online replication.

A later recovered-emulator Wi-Fi pair (deterministic seed 301) used the same
64x64 task-conditioned router in both arms and still scored base `0/1` versus
KV `0/1`, with no invalid actions. The first swipe visibly exposed the Internet
tile, but the following nominal tap returned to the launcher and opened a
long-press menu under software TCG. This is retained as a failed pair, not a
cross-task success; it also leaves policy error and emulator input timing
confounded until the run can be repeated with hardware virtualization.

Cold-booting the software guest did not resolve this: the formerly successful
scripted `swipe/swipe/click(260,200)/click(870,920)` acceptance sequence also
returned zero, with `wifi_on=1`. The ActivityController log contains ANRs from
SystemUI, network stack, Bluetooth, `system_server`, phone, and other services;
a direct ADB tap independently surfaced a system-not-responding dialog. New
online claims must therefore pass the committed KVM/health preflight first.

A lightweight screenshot-conditioned state router is also included. On the
current independent system-task split it reaches `5/6` action-type accuracy
(`3/3` on click states). Its cross-task online result is still pending because
the software-only emulator developed a black-render/system-server failure;
those corrupted probes are explicitly excluded. The committed AndroidWorld
patch now permits a physical-size fallback so action execution does not depend
on a fragile `dumpsys input` call under TCG.

The energy term was also calibrated rather than left at its numerically
negligible initial coefficient. In a contemporaneous 10-epoch comparison,
unregularized Return-KV has held-out margin `0.2209` at mean activation energy
`3.00e-4`; `lambda_energy=3000` retains margin `0.2160` while reducing energy
to `6.72e-5` (4.47x lower). `lambda_energy=100000` reaches `8.19e-7` but drops
margin below the initial checkpoint, demonstrating the expected Pareto
trade-off and providing a practical low-energy setting.

The learned-critic gate is currently negative. An MLP critic trained on 29
system trajectories obtains train AUC `1.0` but only held-out AUC
`0.75–0.8125` with saturated probabilities; a regularized linear critic is
unstable across seeds (`0.375/0.625/0.625`) and underperforms raw trajectory
log-probability ranking (`0.75`). The critic code is retained for
reproduction, but critic-guided KV backprop is deliberately not reported as
an improvement until substantially more independent trajectories exist.

A task/screenshot-conditioned rectified-flow sampler over four-step action
chunks is implemented as a separate exploration module. On two independent
held-out successful trajectories, its mean endpoint MSE is `0.1580` versus
Gaussian noise `1.1474`. After projection to the structured action manifold,
best-of-128 MSE is `0.00247` for Wi-Fi and `0.0924` for Bluetooth. However,
mean behavior-cloning baselines are better than average flow samples, and the
critic gate above cannot yet select the best endpoints reliably. The honest
status is therefore candidate-coverage go, average-policy/critic-chain no-go.

For the current linear KV residual, internalization can be exact rather than
approximate: `y'=(I+AB)y` and `y=Wx` imply a merged projection
`W'=(I+AB)W` (and the same transform for bias). Folding the best checkpoint
into eight V projections removes all inference hooks. The hooked teacher and
merged student have identical held-out Top-1 (`0.5`), MRR (`0.6722`), and
type-conditioned Top-1/MRR (`0.8333/0.8889`); return margin is `0.3767` versus
`0.3836`, with the small difference attributable to bf16 merge rounding.

The K/V ablation is not one-sided. At 10 return-training epochs, K-only moves
held-out margin from `0.1818` to `0.2369` (`+0.0550`), while V-only moves from
`0.1835` to `0.2209` (`+0.0373`). After fork/coordinate shaping, however,
K reaches margin/MRR `0.2818/0.6389` whereas V reaches
`0.3767/0.6722`. This suggests K is stronger for coarse trajectory
separation and V for concrete action/coordinate shaping.

Using SVD to reparameterize the same AITW warm-start makes the rank ablation
comparable. At 10 epochs, ranks `4/8/16` obtain return margins
`0.2094/0.2209/0.2129`; all have Top-1/MRR `0.5/0.6210`. Rank 8 is best for
return separation, rank 16 is already saturated on 41 trajectories, and rank
4 is a reasonable lower-energy budget point.

As a cross-dataset retention check, 16 AITW trajectories have mean action
log-prob `-5.9998` at warm-start, `-6.0107` after AndroidWorld return training,
`-6.0238` with the `lambda_energy=3000` checkpoint, and `-6.1120` after the
strongest fork/coordinate shaping. Pure return training therefore changes the
AITW metric by only about `0.18%`; fork shaping costs about `1.87%`. The
low-energy checkpoint lowers activation energy but does not beat unregularized
return training on this retention metric.

An additional bounded cross-dataset probe uses 32 untouched actions from the
official Mind2Web public training shard, with one positive and four seeded
negative DOM candidates per action. Base and pure AndroidWorld Return-KV both
score Top-1/MRR `0.5000/0.6766`; the mean correct margin changes from
`-0.01402` to `-0.01324`. The low-energy checkpoint scores
`0.4688/0.6609`, and the fork/coordinate merged student scores
`0.4688/0.6557`. Thus pure return training preserves this Web grounding probe
but does not demonstrate a discrete cross-domain gain; stronger shaping causes
a one-example Top-1 regression. This small public-train probe is not presented
as an official Mind2Web benchmark result.

To obtain an online cross-environment check without relying on the unstable
Android software guest, the repository also includes a BrowserGym MiniWoB
adapter. Using BrowserGym `0.14.3`, MiniWoB++ commit
`7fd85d71a4b60325c6585396ec4f48377d049838`, six click/sequence/checkbox/radio/
color tasks, and 24 paired episodes, Base and fork/coordinate KV both succeed
on `22/24`. There are zero improved and zero regressed pairs, and all 24 action
sequences are identical. Mean candidate Top-1 margin changes slightly from
`0.11080` to `0.11072`. One shared failure is a seeded page layout where
Playwright reports that overlapping button TWO intercepts a click on button
ONE; the other is an incorrect visual choice among unlabeled colored boxes.
This is online cross-environment no-regression evidence, not a
cross-environment return improvement.

The first TANGO trajectory-control matrix is recorded in
[`results/tango_trajectory_controls_e5.json`](results/tango_trajectory_controls_e5.json).
On six held-out AndroidWorld trajectories, the shared warm-start margin is
`0.13015`, observed-return training reaches `0.14116`, shuffled return reaches
`0.13219`, and sign-flipped return falls to `0.12312`. Horizons `1/3/5/full`
score `0.13975/0.14387/0.14304/0.14309`. This is useful label-direction
evidence, but not yet the desired trajectory-over-token result: the curve
plateaus after H=3 and first-only is strongest on this tiny split. We therefore
require a larger same-prefix delayed-consequence split before making that
claim.

The expanded MiniWoB click-only power pilot is recorded in
[`results/tango_miniwob_click4_s701_800.json`](results/tango_miniwob_click4_s701_800.json).
Across four families and 100 shared seeds each, Base/Warm/Return/CE/Shuffle
succeed on `329/330/330/333/328` of 400 cases. Return ties the warm start and
trails CE by `0.75pp`; its paired 95% interval versus CE is `[-2.25pp, 0]` and
exact McNemar `p=0.25`. All action differences occur on `click-color`, while
the other families are unchanged or saturated. Return does raise mean
candidate margin to `0.17110` from Base `0.15017`, but this does not convert to
task success. The expanded online gate therefore fails and MiniWoB remains a
diagnostic rather than a positive headline result.

A second MiniWoB gate trains directly on 40 seeded random legal-click
trajectories (19 successful) and evaluates 20 independent trajectories. The
held-out success/failure score margin rises from `0.12792` at the shared
AITW warm start to `0.13989` with return weighting, versus `0.12647` after
shuffling returns and `0.13557` with successful-only action CE. On 30 unseen
`click-color` episodes at the calibrated injection strength `alpha=16`, the
warm start succeeds on `12/30`, shuffled return on `13/30`, and both return
weighting and action CE on `14/30`. Return weighting improves two paired
episodes without regressions, but does not beat action CE online. This passes
the offline objective-specific sanity check, while the online result remains
a weak positive and fails the objective-specificity gate; it is not evidence
of broad cross-task superiority.

The principal 20-epoch run is reproducible with:

```bash
.venv/bin/python -m trajflow_kv.train \
  --config configs/qwen_androidworld.yaml \
  --data-path data/androidworld/multitask_return_train_v2.jsonl \
  --output-dir outputs/gonogo/multitask_v2_warm_return_v_r8_l8_a8_e20 \
  --projector-checkpoint outputs/gonogo/initial_v_l8/kv_projectors.pt \
  --target v --last-n-layers 8 --rank 8 --alpha 8 --epochs 20
```

## Reproducing the later phases

### Same-prefix counterfactual TANGO

The counterfactual data protocol keeps one JSONL row per candidate action at
an immutable GUI prefix, with evaluator-provided `Q`, `V`, and `advantage=Q-V`.
Generate the text-only delayed-consequence smoke data and train the Qwen K/V
projector with the new path as follows:

```bash
.venv/bin/python -m trajflow_kv.tango_advantage \
  --output data/toy/tango_counterfactual.jsonl \
  --seeds 10 --horizon 6

.venv/bin/python -m trajflow_kv.train \
  --config configs/tango_counterfactual_qwen.yaml \
  --counterfactual-data data/toy/tango_counterfactual.jsonl \
  --objective tango \
  --output-dir outputs/tango/counterfactual_qwen
```

`--objective` accepts `tango` (state-conditioned `Q-V` advantage),
`global_return` (same-prefix normalized Q baseline), or `ce` (oracle highest-Q
candidate control). This path scores every candidate under the same text
prefix and temporarily hooks K/V activations; it does not create a hooked-KV
dataset and leaves the existing trajectory `data_path` training unchanged.

### Visual delayed-consequence pilot

The screenshot-backed pilot is a deterministic, emulator-free GUI benchmark
for the return-to-credit gate. `distractor_credit` places harmless X/Y actions
around a hidden A/B fork; `hidden_memory` shows a color cue that disappears
before the choice. Each candidate row keeps the existing
`tango.counterfactual.v1` fields and adds `image`, `history_images`,
`critical_step`, `critical_actions`, `optimal_actions`, and
`is_critical_action`. `history_images` contains the visual observations before
the current decision (including a vanished cue), while `image` is the current
screen. This distinction is essential: a current-screen-only evaluator cannot
test latent visual memory. The task state machine remains directly executable
for online replay.

Generate a small pilot (screenshots are intentionally ignored by git):

```bash
.venv/bin/python scripts/generate_visual_delayed.py \
  --output data/visual_delayed/pilot.jsonl \
  --image-dir data/visual_delayed/images \
  --seeds 10 --horizon 8 --aggregation mean
```

Use the rows with the existing `--counterfactual-data` trainer after verifying
same-prefix fork accuracy. This controlled pilot is intended before spending
compute on AndroidWorld or browser rollouts.

Score every visual candidate with Qwen under the same prefix. The optional
baseline checkpoint is evaluated in the same process, and the output includes
candidate top-1, critical-fork accuracy, per-family results, and non-critical
score changes:

```bash
.venv/bin/python scripts/evaluate_counterfactual_qwen.py \
  --data data/visual_delayed/pilot.jsonl \
  --checkpoint outputs/tango/counterfactual_qwen/kv_projectors.pt \
  --baseline-checkpoint outputs/gonogo/initial_v_l8/kv_projectors.pt \
  --output results/visual_delayed_qwen.json
```

Omit `--checkpoint` to evaluate the zero-residual base VLM. This is an offline
counterfactual ranking diagnostic; it is not an online GUI success result.

The first history-aware P0 rerun is recorded in
[`results/tango_p0_history_memory.json`](results/tango_p0_history_memory.json).
On 400 prefixes (60 critical), warm-start reaches 100% critical-fork accuracy
once the earlier screenshots are actually supplied. One-epoch CE and TANGO
both reach 100% critical-fork accuracy and 100% candidate Top-1 on this small
pilot, so this is a protocol/backbone diagnostic rather than evidence that
TANGO beats CE. Their large non-critical score shifts also motivate the next
state-conditioned memory gate and causal KV-block ablation.

The v2 diagnostic adds a genuinely independent harmless screen between cue
and decision; results are recorded in
[`results/tango_memory_credit_v2_pilot20.json`](results/tango_memory_credit_v2_pilot20.json).
On its 20 hidden-memory critical prefixes, full history scores `1.00`, removing
the cue block falls to `0.35`, and removing the distractor block remains
`1.00`. Decoder ablation localizes the strongest signal to early/middle K and
middle V, with no effect from the late third. A middle-layer state-conditioned
K+V gate preserves the zero-gate Top-1/critical results (`0.9524/1.00`) while
reducing non-critical mean absolute score change from the old fixed
projector's `3.41` to `0.01475` (about 231x). Its cue/distractor mean gates are
`0.1727/0.1508`, so localization is directionally correct but still weak; this
is a stability/mechanism result, not yet a policy-gain result.

Explicit block-level supervision is tracked in
[`results/tango_memory_advantage_pilot20.json`](results/tango_memory_advantage_pilot20.json).
Five critical-only epochs expand the hidden cue/distractor gate gap to `0.443`
(`0.961/0.518`) and raise aggregate candidate Top-1 from `0.9524` to `1.00`,
while critical accuracy remains saturated at `1.00`. A deliberately naive
control that labels every non-critical *action* prefix as zero memory advantage
collapses the gate gap to `0.0083`. This negative control matters: a memory
block may be essential for a later fork even when the current action is
harmless. Consequently, the next training split must use counterfactual
`Q(M)-Q(M^-j)` targets rather than action-critical labels.

The resulting two-stage experiment is summarized in
[`results/tango_memory_credit_two_stage_heldout.json`](results/tango_memory_credit_two_stage_heldout.json).
Qwen policy counterfactuals measure hidden cue/distractor memory advantages of
`+0.0913/-0.0136`. Stage 1 freezes transport and learns gates of
`0.7044/0.0351`; Stage 2 freezes that gate and learns energy-regularized
transport. On disjoint seeds 20–39, aggregate candidate Top-1 improves from
`0.9524` to `0.9857`, hidden-memory Top-1 from `0.8333` to `0.9500`, critical
accuracy stays `1.00`, and non-critical mean absolute score change is only
`0.01683`. These are controlled held-out ranking results over shared task
templates, not yet real-environment online success.

Build state-conditioned preference pairs and train the fork projector from a
return checkpoint:

```bash
.venv/bin/python scripts/make_fork_pairs.py \
  data/androidworld/multitask_return_train_v2.jsonl \
  data/androidworld/system_fork_pairs_v1.jsonl \
  --task-prefix System --max-per-decision 1

.venv/bin/python scripts/make_coordinate_forks.py \
  data/androidworld/multitask_return_train_v2.jsonl \
  data/androidworld/system_coordinate_forks_v1.jsonl \
  --task-prefix System --offsets 100 200 350

.venv/bin/python scripts/train_fork_preferences.py \
  --checkpoint outputs/gonogo/multitask_v2_warm_return_v_r8_l8_a8_e20/kv_projectors.pt \
  --data data/androidworld/system_fork_pairs_v1.jsonl \
  --output-dir outputs/gonogo/return_e20_fork_v1_e5 --epochs 5
```

Run the calibrated low-energy point and evaluate both return separation and
activation transport energy:

```bash
.venv/bin/python -m trajflow_kv.train \
  --config configs/qwen_androidworld.yaml \
  --data-path data/androidworld/multitask_return_train_v2.jsonl \
  --output-dir outputs/gonogo/return_energy3e3_v_r8_l8_a8_e10 \
  --projector-checkpoint outputs/gonogo/initial_v_l8/kv_projectors.pt \
  --target v --last-n-layers 8 --rank 8 --alpha 8 --epochs 10 \
  --lambda-energy 3000

.venv/bin/python scripts/evaluate_return_margin.py \
  --checkpoint outputs/gonogo/return_energy3e3_v_r8_l8_a8_e10/kv_projectors.pt \
  --data data/androidworld/multitask_system_heldout.jsonl \
  --output outputs/gonogo/energy_compare_energy3e3.json \
  --target v --last-n-layers 8 --rank 8 --alpha 8
```

The critic and flow gates are independently reproducible:

```bash
.venv/bin/python scripts/train_trajectory_critic.py \
  --train-data data/androidworld/multitask_return_train_v2.jsonl \
  --train-scores outputs/gonogo/critic_initial_train_scores.json \
  --heldout-data data/androidworld/multitask_system_heldout.jsonl \
  --heldout-scores outputs/gonogo/heldout_margin_initial_a8.json \
  --output-dir outputs/gonogo/linear_critic_s23 --seed 23

.venv/bin/python scripts/train_action_flow.py \
  --train data/androidworld/multitask_success_sft_v2.jsonl \
  --heldout data/androidworld/multitask_system_heldout.jsonl \
  --output-dir outputs/gonogo/action_flow_projected_s31 \
  --epochs 3000 --samples 128 --seed 31
```

Finally, fold the trained KV residual into ordinary model weights and evaluate
the student with no hooks:

```bash
.venv/bin/python scripts/merge_projector_checkpoint.py \
  --checkpoint outputs/gonogo/return_fork_click_e3_coord_e2/kv_projectors.pt \
  --output outputs/gonogo/internalized_return_fork_coord/merged_weights.pt \
  --target v --last-n-layers 8 --rank 8 --alpha 8

.venv/bin/python scripts/evaluate_action_ranking.py \
  --merged-checkpoint outputs/gonogo/internalized_return_fork_coord/merged_weights.pt \
  --data data/androidworld/multitask_system_heldout.jsonl \
  --output outputs/gonogo/internalized_return_fork_coord/heldout_ranking.json
```

Download the smallest official Mind2Web training shard, build the fixed
candidate fixture, and reproduce the cross-dataset probe (raw data stays
gitignored):

```bash
./scripts/download_mind2web_sample.sh
.venv/bin/python scripts/prepare_mind2web.py \
  data/mind2web_raw/train_10.json data/mind2web_eval.jsonl \
  --limit 32 --negatives 4 --seed 17

.venv/bin/python scripts/evaluate_mind2web_ranking.py \
  --data data/mind2web_eval.jsonl \
  --checkpoint outputs/gonogo/multitask_v2_warm_return_v_r8_l8_a8_e10/kv_projectors.pt \
  --output outputs/gonogo/mind2web_return.json
```

Set up the pinned MiniWoB environment and reproduce the online paired probe:

```bash
.venv/bin/pip install '.[browser]'
.venv/bin/playwright install chromium
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git /root/miniwob-plusplus
git -C /root/miniwob-plusplus checkout 7fd85d71a4b60325c6585396ec4f48377d049838
export MINIWOB_URL='file:///root/miniwob-plusplus/miniwob/html/miniwob/'
./scripts/run_miniwob_gate.sh
```

## What is and is not implemented

- Implemented: online KV hooks (no cached activation dataset), K/V/both
  injection, REINFORCE-style trajectory loss, per-task baseline, normalized
  advantages, energy and orthogonality regularization, JSONL trajectory
  format, checkpoints, fork/coordinate preference training, hierarchical
  candidate ranking, loop guards, and smoke tests.
- Adapter-ready: AndroidWorld rollouts can be exported into the same JSONL
  schema using `scripts/androidworld_to_jsonl.py`.
- Implemented and gated: rectified-flow action chunks, learned trajectory
  critic, screenshot/instruction state routing, and hook-free weight folding.
  Flow passes candidate-coverage but not average-policy selection; the critic
  is a data-limited no-go; exact weight folding passes the internalization
  equivalence check.

## Quick start (no model download)

```bash
cd /root/trajflow-kv
python3 -m venv .venv
.venv/bin/pip install -e '.[test]'
.venv/bin/pytest -q
.venv/bin/python -m trajflow_kv.train --config configs/toy.yaml
```

## Real model and data

```bash
.venv/bin/pip install -e '.[train]'
./scripts/setup_cuda126.sh
./scripts/download_model.sh
./scripts/download_aitw_sample.sh 16
.venv/bin/python -m trajflow_kv.train --config configs/qwen_aitw.yaml
```

The Qwen run requires substantially more memory than the toy test. On a 16GB
GPU use batch size 1, gradient accumulation, gradient checkpointing and a
small image pixel budget. The backbone is frozen; only projectors train.
`max_trajectories: 1` is the committed real-model smoke-test default; increase
it only after observing peak GPU memory on your machine.

Verified on an RTX A4000 (16GB): 16 viewer examples use 9.28 GiB peak allocated
GPU memory, hook all 72 K/V projections across 36 language layers, and train
294,912 projector parameters. Run the verified batch with
`--max-trajectories 16`.

## Trajectory JSONL schema

Each line represents one trajectory:

```json
{"task_id":"clock-1","instruction":"Set an alarm","return":1.0,"steps":[{"image":"/abs/s0.png","history":[],"action":"{\"action_type\":\"click\",\"x\":0.5,\"y\":0.8}"}]}
```

`return` is the environment return. Every action log-probability in a
trajectory receives the same normalized trajectory advantage in this minimal
version. Later work can replace this with reward-to-go or a critic.

The bundled AITW viewer sample contains successful demonstrations only, so
`lambda_action` supplies an imitation warm-start. Pure return optimization
requires mixed-return rollouts from AndroidWorld; set `lambda_action: 0` for
that experiment. Treating all-success demonstrations as policy-gradient data
would yield zero centered advantage and is intentionally avoided.

## Reproducibility notes

- Model: `Qwen/Qwen2.5-VL-3B-Instruct`, pinned by the snapshot downloader.
- Online benchmark: AndroidWorld; its emulator is intentionally not bundled.
- Offline source: AITW. The helper downloads a bounded sample/index rather
  than the multi-million-step corpus.
- Never commit Hugging Face or GitHub tokens. `.env`, model files, datasets,
  checkpoints and logs are ignored.

## AndroidWorld mixed-return loop

AndroidWorld's official Docker image exposes a FastAPI server on port 5000.
It can run on the same host or a separate emulator host; this repository only
needs HTTP access to it. Collect repeated stochastic rollouts for one fixed
task instance so the task-group baseline is meaningful:

```bash
.venv/bin/python scripts/collect_androidworld.py \
  --server-url http://ANDROID_HOST:5000 \
  --task OpenAppTaskEval --task-index 0 \
  --rollouts 8 --max-steps 1 --temperature 1.5 \
  --output data/androidworld/open_app_rollouts.jsonl

.venv/bin/python scripts/validate_rollouts.py data/androidworld/open_app_rollouts.jsonl
.venv/bin/python -m trajflow_kv.train --config configs/qwen_androidworld.yaml
```

For a paired check, collect baseline and candidate rollouts with the same
`--seed`, task, horizon, and temperature, then run:

```bash
.venv/bin/python scripts/compare_rollouts.py \
  --baseline data/androidworld/eval_pre.jsonl \
  --candidate data/androidworld/eval_post.jsonl \
  --output outputs/qwen-androidworld-return/controlled_eval.json
```

The summary reports paired improvements/regressions, success-rate delta, and
invalid-action counts. Small smoke samples are a pipeline check, not evidence
of a statistically significant policy improvement.

The collector uses the official `/screenshot`, `/execute_action`, task
initialize/tear-down, goal, and score endpoints. Screenshots and canonical
actions are saved per step. Invalid model JSON becomes a recorded `wait`
action rather than crashing or silently dropping the trajectory. AITW-style
point coordinates and two-point swipes are converted to AndroidWorld's action
schema at this boundary.

Pure return training (`lambda_action: 0`) refuses to run if every task group
has constant return. This prevents an all-success or all-failure batch from
silently producing zero centered policy-gradient signal. The AndroidWorld
config loads the AITW projector checkpoint before applying return updates.

### Running without `/dev/kvm`

The verified software-emulation setup is reproducible with:

```bash
./scripts/setup_androidworld_software.sh
./scripts/run_androidworld_software.sh
```

The setup pins AndroidWorld commit `3e508885`, installs API 33/Pixel 6 and a
Python 3.11 environment, and applies `patches/androidworld-tcg.patch`. The patch
makes ADB timeouts configurable, allows screenshot-only operation without the
accessibility forwarder, and makes first-time app setup optional. For smoke
tasks it also enables lightweight task initialization, skipping date and app
snapshot resets that are not required by `OpenAppTaskEval`. Do not use that
mode for tasks whose evaluator depends on restored app state. The runner uses
`-accel off`, increases Android's watchdog timeout multiplier, retries an
Android ActivityController during boot, skips Setup Wizard, and serves the
official API on `127.0.0.1:5000`.

On the reference RTX A4000 container, first boot took roughly 25 minutes.
Native AndroidWorld initialization then returned a 1080x2400 screenshot in
15.4 seconds. A real `ContactsAddContact` initialize/score/teardown cycle and
a Qwen + KV checkpoint rollout were both verified.

Software TCG remains useful for API smoke tests but is no longer accepted for
the multi-seed evidence gate. On a KVM-capable host, run the preflight and
paired gate as follows:

```bash
.venv/bin/python scripts/preflight_androidworld.py --require-kvm \
  --output outputs/gonogo/androidworld_preflight.json

.venv/bin/python scripts/run_paired_online_gate.py \
  --task SystemWifiTurnOff --task-index 0 --rollouts 3 --seed 301 \
  --max-steps 4 \
  --checkpoint outputs/gonogo/return_fork_click_e3_coord_e2/kv_projectors.pt \
  --state-router-checkpoint outputs/gonogo/state_router64_taskcond_s17/state_router.pt \
  --output-dir outputs/gonogo/wifi_kvm_gate_s301
```

The gate uses the same seeds and structured policy in both arms and repeats
preflight between Base and KV. This prevents a guest destabilized by the Base
arm from silently contaminating only the KV arm.

To exercise the exact return path without an emulator, create a small mixed
fixture from the downloaded sample:

```bash
.venv/bin/python scripts/make_mixed_return_smoke.py \
  data/aitw/train.jsonl /tmp/trajflow-mixed.jsonl
.venv/bin/python scripts/validate_rollouts.py /tmp/trajflow-mixed.jsonl
.venv/bin/python -m trajflow_kv.train \
  --config configs/qwen_androidworld.yaml \
  --data-path /tmp/trajflow-mixed.jsonl \
  --output-dir /tmp/trajflow-return-smoke
```
