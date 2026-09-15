# TANGO experiment contract

TANGO is evaluated as a trajectory-policy method, not as a generic KV
projector. Every comparison must use the same task, seed, initial state, action
parser, step budget, candidate set, and backbone checkpoint.

## Claims and falsification tests

1. **Return is written into memory.** Compare TANGO with the shared warm start,
   all-action CE, successful-only CE, shuffled return, and sign-flipped return.
   Report return margin, success/failure AUC, MRR, and Top-1. A useful result
   requires TANGO to beat shuffled return in every tested domain without a
   Top-1 collapse.
2. **The signal is trajectory consequence, not token correctness.** Hold the
   parameter and data budgets fixed while varying `H=1/3/5/full`, training on
   only the first or final step, and removing history. Evaluate same-prefix
   forks using fork accuracy, positive-minus-negative action margin, and the
   return obtained after executing the selected branch.
3. **Online task success improves.** AndroidWorld is the primary benchmark and
   MiniWoB is the first independent browser benchmark. Report paired deltas,
   task-clustered bootstrap confidence intervals, exact McNemar tests,
   per-task wins/losses/ties, illegal actions, steps, and latency.
4. **The policy can be internalized.** Distill candidate-set teacher
   probabilities from the hooked TANGO teacher into a Qwen2.5-VL-3B LoRA
   student (`r=16`, `alpha=32`, q/k/v/o projections). The no-hook student must
   recover at least 70% of the teacher's online gain over CE and outperform a
   matched CE-LoRA student.

## Execution gates

AndroidWorld is forbidden until a hardware-accelerated host passes 20 complete
setup/action/evaluation/teardown cycles with 20/20 correct deterministic
scores, no ANR, and no system-server or network-stack crash. The current host
has no `/dev/kvm` and is therefore not accepted.

After preflight, run an 8-task x 3-seed pilot before locking the 40-task x
5-seed main suite. The main result passes only if TANGO-Energy exceeds action
CE by at least five percentage points, the task-clustered 95% interval excludes
zero, exact McNemar `p < 0.05`, and illegal actions do not increase.

For MiniWoB, first run the supported click-only four-family x 100-seed pilot.
Text entry, login, list selection, and autocomplete require an action executor
extension before the proposed eight-family experiment is valid. A main-claim
result requires a three-to-five point advantage over CE with both paired tests
passing; otherwise MiniWoB remains a sanity check.

## Resource policy

Qwen2.5-VL-3B is the primary model. Qwen2.5-VL-7B is a scale ablation only
after the primary online gate passes. Full AITW, VisualWebArena, OSWorld, and
UI-TARS downloads are deferred because they do not resolve the current
evidence bottleneck and exceed the safe budget of this 65GB workspace.
ScreenSpot is retained strictly as a single-step grounding retention test.

