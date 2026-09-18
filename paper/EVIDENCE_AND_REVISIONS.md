# SIGMA — source ledger and substantive revisions

Prepared for the single-title, single-`main.tex` ICLR 2027 rewrite. This is an author-side provenance file, not part of the anonymous paper text.

## 1. What the manuscript claims

The strongest supported statement is conditional and concrete: matched replacement of particular historical decoder key spans changes a frozen VLM's current decision, with opposing signed effects for superseded and latest updates in the reported controlled histories. The learned SIGMA system uses these measurements as explicit supervision for a history-aware controller and a separately optimized low-rank correction.

The current evidence does **not** establish recurrent SIGMA's superiority over CE on MemGUI-Bench, AndroidWorld, or WorkArena++. Those fields are `NR`. The manuscript does not claim to solve optimal transport, prove that gate signs imply beneficial key scaling, or reproduce the full SPD pipeline with a simplified baseline.

## 2. Primary experiment sources

User-supplied experiment reports:
- `Pasted text(20260827-023458).txt`
- `Pasted text(20260828-081909).txt`

Inspected repository: `https://github.com/Yunbo-max/trajflow-kv`
Snapshot used for interpretation: `131014ab50d6bbccff4c5ecb1332679256fa40b7`.
This rewrite did not modify or push the repository and did not run GPU training or GUI episodes.

| Manuscript object | Source | Scope |
|---|---|---|
| Figure 1: 60→100/0/60 pilot | `results/tango_v4_interference_chain_kv_credit_pilot10.json` | 10-prefix, same-forward matched K patch, layers 12–23; current decision accuracy and mean action-score effects |
| Table 1: 200-prefix sign pattern | `results/tango_v2_teacher_credit200.json` and supplied aggregate report | 1,000 blocks across 200 prefixes; candidate-value proxy, not native GUI return |
| Candidate-value definition | `scripts/estimate_memory_advantages_qwen.py`, `expected_continuation_return` | Softmax over candidate scores, multiplied by fixed row Q-values |
| Matched donor aggregation | `scripts/estimate_kv_memory_credit_qwen.py` | Mean of original-minus-patched candidate values; two reference donors or one other reference for a reference target |
| 400 paired MiniWoB cases | `results/tango_miniwob_click4_s701_800.json` and supplied report | Earlier global-return policy; 4 task families × 100 common seeds |
| Horizon controls | `results/tango_trajectory_controls_e5.json` | Six held-out AndroidWorld trajectories; offline margin only |
| Earlier 60/60 memory trainer | `results/tango_memory_transport_stage2_critical.json`, `..._full.json`, `..._test20_39.json` | Different controlled set; not a matched recurrent-SIGMA comparison |
| Policy and residual interpretation | `trajflow_kv/projector.py`, `trajflow_kv/qwen_policy.py` | Distinguish static linear residual, history-dependent gating, and old task-specific candidate policy |

The paper's role means and sign fractions reproduce the supplied aggregate report. The code and sampled raw-record structure were independently inspected to establish the metric semantics; this rewrite does not claim a fresh recomputation of every large-JSON aggregate.

## 3. Critical corrections relative to the previous draft

### Separate the two measured utilities

The ten-prefix result measures mean teacher-forced correct-action token log-score. The 200-prefix teacher file computes a finite-candidate value proxy:

`softmax(candidate_scores / temperature) dot candidate_Q_values`.

These are **different units and different estimands**. The rewritten paper gives separate equations and labels, and does not compare their numerical magnitudes on a shared axis. Native environment return is a third, explicitly separate utility.

### Causal credit is not an intrinsic good/bad label

An effect is conditional on the prefix, donor, layer set, and chosen utility. A reference's near-zero average effect does not prove per-instance neutrality. A semantically old record is not assigned negative credit by definition. The prior pilot's weakly positive Initial effect and the larger set's mixed Initial signs are preserved.

### Same-forward patch versus clean donor extraction

The reported pilot uses **same-forward, in-sequence donors**. The proposed training extractor freezes donor tensors from a clean prefix prefill. The manuscript names these separately; it does not retrospectively relabel legacy observations as clean-donor experiments. Subsequent activations can change after patching; only the externally fixed input/target positions and the specified intervention site are held fixed.

### Gate sign is not a proof of suppression

The original `K + g U U^T K` notation was too loosely interpreted. With tokens as rows, the implemented class of update is a general low-rank residual:

`X' = X + (alpha/r) g X B A`.

The map is not an orthogonal projector unless constrained. A negative gate does not universally lower a block's attention contribution. SIGMA's policy loss learns the correction direction; measured signed credit supplies supervision. The paper and figure prompts now make this distinction explicit.

### No invented policy-gradient equivalence

On fixed logged trajectories, return-weighted likelihood is an offline surrogate. It is not automatically an unbiased gradient of current-policy expected return. The appendix defines the reference-policy-weighted action-advantage comparator and distinguishes it from the historical unweighted enumeration.

### No negative KL weights or dynamic-gate merge claim

The optional no-hook student uses nonnegative state weights. Residual logits are centered to remove arbitrary common shifts. Static linear merging is an engineering identity; it is not evidence that an input-dependent recurrent gate has been distilled away.

### Real GUI evaluation remains the application test

Interference Chain is a causal diagnostic. MemGUI-Bench, AndroidWorld and WorkArena++ retain distinct task definitions, denominators and evaluators. Both mobile environments need a healthy Android runtime; MemGUI is not a way around absent KVM. WorkArena++ needs an actual service instance. Full-suite external transfer excludes all target-suite tasks/trajectories from training and selection; a later in-domain split must be named separately.

## 4. Writing skills actually consulted

ARIS repository:
`https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep`

Read:
`skills/skills-codex/paper-write/SKILL.md`

ARIS links the anti-defensive writing repository:
`https://github.com/Adkid-Zephyr/anti-defensive-writing-Skill`

Read:
`skills/anti-defensive-writing-en/SKILL.md`

Applied: strongest defensible observation first; a single scientific argument instead of a chronological lab report; every experiment has a specific explanatory role; concise limitations; concrete sentences and verified citations. Instructions that would hide relevant negative evidence or turn missing results into asserted wins were not adopted.

The user's previous `Scientific_Figure_Style_Atlas.md` was retrieved and read. The figure brief follows its fact locks, training/inference separation, and three genuinely different visual organizations per figure. It does not claim that additional named figure skills were found when only this atlas was available.

## 5. Verified literature and role in the manuscript

Metadata was checked on the primary pages below. Bibliographic entries are embedded in `main.tex` rather than split into another source file.

1. Bai et al. (2025), *Qwen2.5-VL Technical Report*. https://arxiv.org/abs/2502.13923 — backbone.
2. Boisvert et al. (2024), *WorkArena++: Towards Compositional Planning and Reasoning-based Common Knowledge Work Tasks*. https://arxiv.org/abs/2407.05291 — browser workflow evaluation.
3. Darcet et al. (2024), *Vision Transformers Need Registers*. https://arxiv.org/abs/2309.16588 — phenomenon-to-remedy writing structure and related internal visual-representation analysis.
4. Hao et al. (2026), *Self-Policy Distillation via Capability-Selective Subspace Projection*. https://arxiv.org/abs/2605.22675 — capability-selective generation/distillation comparison. The title is not “Self-Play Distillation.”
5. Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*. https://arxiv.org/abs/2106.09685 — optional student parameterization.
6. Liu et al. (2026), *MemGUI-Bench: Benchmarking Memory of Mobile GUI Agents in Dynamic Environments*. https://arxiv.org/abs/2602.06075 — memory-specific real mobile benchmark.
7. Rawles et al. (2023), *Android in the Wild: A Large-Scale Dataset for Android Device Control*. https://arxiv.org/abs/2307.10088 — external demonstration source.
8. Rawles et al. (2024), *AndroidWorld: A Dynamic Benchmarking Environment for Autonomous Agents*. https://arxiv.org/abs/2405.14573 — native mobile evaluation. Not the invented “Zhang et al. 2026” attribution in the earlier draft.
9. Wang et al. (2024), *Agent Workflow Memory*. https://arxiv.org/abs/2409.07429 — external procedural memory.
10. Wu et al. (2024), *ReFT: Representation Finetuning for Language Models*. https://arxiv.org/abs/2404.03592 — representation adaptation.
11. Zhang and Nanda (2024), *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*. https://arxiv.org/abs/2309.16042 — donor/metric sensitivity. Authors are Fred Zhang and Neel Nanda.
12. Zhang et al. (2026), *FocusMem: Factorizing Content, Readout, and Trust in Latent GUI Memory*. https://arxiv.org/abs/2608.04530 — close latent-memory competitor. The previous “Arora et al.” citation was incorrect.

Full text inspected for organization/mechanism: SPD, FocusMem, and Vision Transformers Need Registers. The manuscript is original prose; these sources guide structure and positioning, not copied paragraphs.

## 6. What remains to replace before submission

- Fill the native real-GUI result table from completed paired episodes.
- Add matched ablation outcomes; the current ablation table is a hypothesis/measurement design, not results.
- Measure credit→return agreement with actual restored-state rollouts.
- Report controller generalization, per-instance neutrality, runtime, and training-seed uncertainty.
- Run optional no-hook policy distillation only once the teacher gain exists.
- Complete the authors' factual AI-use verification statement and provide an anonymized code/data snapshot.
- Rebuild after adding any result figure and reconfirm the main-text nine-page limit.
