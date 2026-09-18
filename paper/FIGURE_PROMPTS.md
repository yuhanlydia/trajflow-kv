# SIGMA — figure production briefs

Paper: **SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents**

## 使用方法

这里给出 8 张图的制作规格，每张图有 **A / B / C 三个可独立使用的完整构图方案**。任选一种，不要把三种硬拼在一张图里。F1、F2 已在 `main.tex` 中用可编辑矢量代码实现；其他图是后续实验完成后的绘图规格。所有定量结果图应由真实 CSV/JSON 用绘图程序生成，图像生成模型仅用于布局与示意元素，不负责补造数据。

**正文优先保留 F1（现象）和 F2（机制）。** F3 可替换正文的 200-prefix 表，不能重复占用版面；F4 可在结果齐备后替换主结果表的一部分。F5–F8 以附录为主，正文只选择一张最有解释力的图，届时重新检查九页限制。

事实标记：`MEASURED` 为已有记录；`DERIVED` 为由已有记录直接计算；`CONCEPTUAL` 为方法示意；`MISSING` 为尚未提供实测数据。示意 GUI 不得冒充实际截图。历史 block 的正负含义必须写成 **original minus matched replacement**；不能把正值写成“替换会提高性能”。

统一视觉约定：白底、紧凑科学排版、无标题横幅、无大脑/机器人装饰、无霓虹或立体卡片。可用低饱和青绿、赭红、灰色表示 positive / negative / reference，但必须同时用 `+ / − / ≈0`、线型或描边编码。缩至论文 5.5-inch 栏宽后，所有文字至少约 7–8 pt 可读。箭头有且仅有明确的数据流、控制流或梯度含义。


---

## F1 — The motivating phenomenon: a stored update can interfere

**Evidence:** MEASURED + CONCEPTUAL  
**Placement:** Main text, Figure 1 (already drawn in main.tex).

### Data contract / 事实锁

Use exactly this measured ten-prefix diagnostic: frozen Qwen2.5-VL-3B-Instruct, K-only intervention, zero-based decoder layers 12–23. Original critical-decision accuracy is 6/10 = 60%. Replace Initial with reference: 5/10 = 50%, mean action-score effect +0.046857. Replace Reference 1: 6/10 = 60%, effect −0.019419. Replace Superseded Update: 10/10 = 100%, effect −0.096451. Replace Reference 2: 6/10 = 60%, effect +0.001345. Replace Latest Update: 0/10 = 0%, effect +0.236780. Define effect as original minus replacement. The history sequence is Initial → Reference → Superseded Update → Reference → Latest Update → Current Decision with eight neutral slots. Inputs and recipient token positions are unchanged. These are intervention results, not performance after training SIGMA. No error bars or significance stars are available.

### A — Timeline and three intervention contrasts

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
Use exactly this measured ten-prefix diagnostic: frozen Qwen2.5-VL-3B-Instruct, K-only intervention, zero-based decoder layers 12–23. Original critical-decision accuracy is 6/10 = 60%. Replace Initial with reference: 5/10 = 50%, mean action-score effect +0.046857. Replace Reference 1: 6/10 = 60%, effect −0.019419. Replace Superseded Update: 10/10 = 100%, effect −0.096451. Replace Reference 2: 6/10 = 60%, effect +0.001345. Replace Latest Update: 0/10 = 0%, effect +0.236780. Define effect as original minus replacement. The history sequence is Initial → Reference → Superseded Update → Reference → Latest Update → Current Decision with eight neutral slots. Inputs and recipient token positions are unchanged. These are intervention results, not performance after training SIGMA. No error bars or significance stars are available.

COMPOSITION
Use a wide 2.9:1 landscape canvas. Allocate the upper 40% to a single continuous five-block history timeline and the current decision. The history tiles are schematic interface thumbnails with short labels, not invented screenshots. Allocate the lower 60% to three equal, aligned contrasts: “Replace superseded” 60% → 100%; “Replace latest” 60% → 0%; “Replace reference” 60% → 60%. Under each put the original-memory score effect: −0.0965, +0.2368, approximately zero (both exact reference values belong in the caption). The visual center is the opposing direction of the first two arrows. Add the short message “Same screenshots; different historical keys.” Explicitly mark n=10 paired prefixes. Keep the model as one small frozen box, not a network illustration.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Matched historical-key interventions produce opposing decision effects. In ten controlled prefixes, replacing a superseded update raises critical-decision accuracy from 60% to 100%, while replacing the latest update reduces it to 0%; reference replacements leave accuracy at 60%. All inputs and recipient token positions are held fixed. Score effects are mean target-action log-score differences, original minus replacement. History tiles are schematic.
```

### B — Counterfactual intervention ladder

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
Use exactly this measured ten-prefix diagnostic: frozen Qwen2.5-VL-3B-Instruct, K-only intervention, zero-based decoder layers 12–23. Original critical-decision accuracy is 6/10 = 60%. Replace Initial with reference: 5/10 = 50%, mean action-score effect +0.046857. Replace Reference 1: 6/10 = 60%, effect −0.019419. Replace Superseded Update: 10/10 = 100%, effect −0.096451. Replace Reference 2: 6/10 = 60%, effect +0.001345. Replace Latest Update: 0/10 = 0%, effect +0.236780. Define effect as original minus replacement. The history sequence is Initial → Reference → Superseded Update → Reference → Latest Update → Current Decision with eight neutral slots. Inputs and recipient token positions are unchanged. These are intervention results, not performance after training SIGMA. No error bars or significance stars are available.

COMPOSITION
Use a wide 2.4:1 canvas. Left 35%: stack five labeled historical key spans in a decoder band with a fixed current-decision query. Right 65%: a vertically aligned intervention ladder with rows Original, Patch superseded, Patch latest, Patch reference. Each row shows exactly one highlighted changed span and a right-aligned accuracy endpoint, 60%, 100%, 0%, 60%. Keep positions of all other spans aligned, making the intervention target visible. Place donor replacement arrows only on the patched rows. Put “original − replacement” beside the score legend. Do not imply the internal states downstream of the intervention are unchanged. No arrows from ground-truth labels into the model input.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Matched historical-key interventions produce opposing decision effects. In ten controlled prefixes, replacing a superseded update raises critical-decision accuracy from 60% to 100%, while replacing the latest update reduces it to 0%; reference replacements leave accuracy at 60%. All inputs and recipient token positions are held fixed. Score effects are mean target-action log-score differences, original minus replacement. History tiles are schematic.
```

### C — Paired-endpoint evidence panel

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
Use exactly this measured ten-prefix diagnostic: frozen Qwen2.5-VL-3B-Instruct, K-only intervention, zero-based decoder layers 12–23. Original critical-decision accuracy is 6/10 = 60%. Replace Initial with reference: 5/10 = 50%, mean action-score effect +0.046857. Replace Reference 1: 6/10 = 60%, effect −0.019419. Replace Superseded Update: 10/10 = 100%, effect −0.096451. Replace Reference 2: 6/10 = 60%, effect +0.001345. Replace Latest Update: 0/10 = 0%, effect +0.236780. Define effect as original minus replacement. The history sequence is Initial → Reference → Superseded Update → Reference → Latest Update → Current Decision with eight neutral slots. Inputs and recipient token positions are unchanged. These are intervention results, not performance after training SIGMA. No error bars or significance stars are available.

COMPOSITION
Use a 2.5:1 landscape canvas. Upper 25%: compact labeled history sequence. Lower left 35%: schematic decoder key span replacement before positional encoding. Lower right 65%: a paired-endpoint plot on a single 0–100% horizontal accuracy axis. Each treatment starts at the same 60% baseline and ends at 50%, 60%, 100%, 60%, or 0% respectively for the five historical blocks. Label the endpoints directly; these are five aggregate paired conditions over ten prefixes, not ten fabricated individual lines. An inset states “Patch superseded repairs 4 baseline errors; patch latest loses 6 baseline correct decisions.” Those counts are derived from the supplied accuracy totals. No population-confidence claim.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Matched historical-key interventions produce opposing decision effects. In ten controlled prefixes, replacing a superseded update raises critical-decision accuracy from 60% to 100%, while replacing the latest update reduces it to 0%; reference replacements leave accuracy at 60%. All inputs and recipient token positions are held fixed. Score effects are mean target-action log-score differences, original minus replacement. History tiles are schematic.
```

### Caption

Matched historical-key interventions produce opposing decision effects. In ten controlled prefixes, replacing a superseded update raises critical-decision accuracy from 60% to 100%, while replacing the latest update reduces it to 0%; reference replacements leave accuracy at 60%. All inputs and recipient token positions are held fixed. Score effects are mean target-action log-score differences, original minus replacement. History tiles are schematic.

### Required files / next data

Existing pilot result: `results/tango_v4_interference_chain_kv_credit_pilot10.json`. No additional experiment is needed to redraw this figure.

---

## F2 — SIGMA separates measurement, prediction, and correction

**Evidence:** CONCEPTUAL, aligned with the specified method  
**Placement:** Main text, Figure 2 (already drawn in main.tex).

### Data contract / 事实锁

The backbone is frozen. Training-time matched historical KV interventions yield donor-relative signed credit c_j^u = u(M) − mean_d u(Patch_j,d(M)). Utility type is explicitly recorded: action score, candidate value, or native rollout return; they are not mixed. Clean prefix-only block features and a current-decision feature feed a small history-aware controller. The controller scans the observed history with a GRU and predicts each block’s credit using that block, the final history state, and the current decision feature. Predicted credit sets g_j=tanh(c_hat_j/T_g). A learned residual changes historical projections as X'_j=X_j+(alpha/r)g_j X_j B A. B A is a general low-rank map, not an orthogonal projector by default. The action loss trains the correction direction; credit loss trains the prediction and energy regularization bounds the residual. At deployment no gold action, true return, donor label, or per-block intervention sweep is provided. A clean feature prefill and an adapted policy pass are both charged to runtime. Optional no-hook policy distillation is not required for this central pipeline.

### A — Training/deployment swimlanes

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
The backbone is frozen. Training-time matched historical KV interventions yield donor-relative signed credit c_j^u = u(M) − mean_d u(Patch_j,d(M)). Utility type is explicitly recorded: action score, candidate value, or native rollout return; they are not mixed. Clean prefix-only block features and a current-decision feature feed a small history-aware controller. The controller scans the observed history with a GRU and predicts each block’s credit using that block, the final history state, and the current decision feature. Predicted credit sets g_j=tanh(c_hat_j/T_g). A learned residual changes historical projections as X'_j=X_j+(alpha/r)g_j X_j B A. B A is a general low-rank map, not an orthogonal projector by default. The action loss trains the correction direction; credit loss trains the prediction and energy regularization bounds the residual. At deployment no gold action, true return, donor label, or per-block intervention sweep is provided. A clean feature prefill and an adapted policy pass are both charged to runtime. Optional no-hook policy distillation is not required for this central pipeline.

COMPOSITION
Use a 3.1:1 landscape figure with two horizontal lanes. Upper lane, 45% height: Observed GUI prefix → Frozen VLM / clean KV → Matched intervention → Signed credit labels. Lower lane, 55%: Prefix-only pooled features → History-aware controller → Signed gate + learned residual → GUI action → Environment. A dashed downward arrow from signed labels goes only to the controller loss. A separate policy-supervision arrow enters the learned residual during training. Put a compact boundary note “Labels: training only” and “Deployment: predicted credit.” The dominant object is the middle controller/residual pair. Do not draw a fast/slow continual-learning loop or a diffusion process. Color should encode teacher supervision versus deployed computation, not success versus failure.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
SIGMA learns to predict intervention-derived memory credit and uses it to condition historical KV corrections. The teacher compares original and matched replacement computations. A prefix-only recurrent controller predicts signed effects, and policy supervision learns the residual direction under an energy penalty. Deployed control uses predicted labels rather than an intervention search.
```

### B — Measurement-to-control lifecycle

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
The backbone is frozen. Training-time matched historical KV interventions yield donor-relative signed credit c_j^u = u(M) − mean_d u(Patch_j,d(M)). Utility type is explicitly recorded: action score, candidate value, or native rollout return; they are not mixed. Clean prefix-only block features and a current-decision feature feed a small history-aware controller. The controller scans the observed history with a GRU and predicts each block’s credit using that block, the final history state, and the current decision feature. Predicted credit sets g_j=tanh(c_hat_j/T_g). A learned residual changes historical projections as X'_j=X_j+(alpha/r)g_j X_j B A. B A is a general low-rank map, not an orthogonal projector by default. The action loss trains the correction direction; credit loss trains the prediction and energy regularization bounds the residual. At deployment no gold action, true return, donor label, or per-block intervention sweep is provided. A clean feature prefill and an adapted policy pass are both charged to runtime. Optional no-hook policy distillation is not required for this central pipeline.

COMPOSITION
Use three connected columns of unequal width, 28% / 32% / 40%, on a 2.7:1 canvas. Column 1 “Measure”: original and patched KV branches converge into a signed difference. Column 2 “Distill credit”: five block embeddings feed an ordered recurrent strip; each output reads both its own block and final recurrent state. Column 3 “Adapt”: predicted positive/negative/near-zero values condition low-rank residuals within a frozen decoder and produce one executable action. Draw the current task feature as a thin shared conditioning line only to legitimate prefix-dependent modules. Show both credit regression and policy training, with solid data arrows and dashed gradient arrows. The phrase “sign is supervision, not a guaranteed correction” can be a small annotation beside the residual.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
SIGMA learns to predict intervention-derived memory credit and uses it to condition historical KV corrections. The teacher compares original and matched replacement computations. A prefix-only recurrent controller predicts signed effects, and policy supervision learns the residual direction under an energy penalty. Deployed control uses predicted labels rather than an intervention search.
```

### C — Internal mechanism-centered view

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
The backbone is frozen. Training-time matched historical KV interventions yield donor-relative signed credit c_j^u = u(M) − mean_d u(Patch_j,d(M)). Utility type is explicitly recorded: action score, candidate value, or native rollout return; they are not mixed. Clean prefix-only block features and a current-decision feature feed a small history-aware controller. The controller scans the observed history with a GRU and predicts each block’s credit using that block, the final history state, and the current decision feature. Predicted credit sets g_j=tanh(c_hat_j/T_g). A learned residual changes historical projections as X'_j=X_j+(alpha/r)g_j X_j B A. B A is a general low-rank map, not an orthogonal projector by default. The action loss trains the correction direction; credit loss trains the prediction and energy regularization bounds the residual. At deployment no gold action, true return, donor label, or per-block intervention sweep is provided. A clean feature prefill and an adapted policy pass are both charged to runtime. Optional no-hook policy distillation is not required for this central pipeline.

COMPOSITION
Use a 2.4:1 canvas. Center 55%: a large simplified decoder with historical visual KV blocks, current-screen tokens, and action tokens; highlight only historical blocks as editable. Left 25%: a paired original/matched-replacement teacher produces signed credit. Above the decoder: the recurrent credit controller, receiving pooled historical features and clean query context. Right 20%: action execution and a small output policy box. Add the equation X'=X+(alpha/r)gXBA as the single principal formula. Current screen and target action must be outside the historical intervention mask. Avoid implying that K-only findings prove all failures are attention retrieval failures. Make teacher measurement and deployed correction visibly distinct operators.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
SIGMA learns to predict intervention-derived memory credit and uses it to condition historical KV corrections. The teacher compares original and matched replacement computations. A prefix-only recurrent controller predicts signed effects, and policy supervision learns the residual direction under an energy penalty. Deployed control uses predicted labels rather than an intervention search.
```

### Caption

SIGMA learns to predict intervention-derived memory credit and uses it to condition historical KV corrections. The teacher compares original and matched replacement computations. A prefix-only recurrent controller predicts signed effects, and policy supervision learns the residual direction under an energy penalty. Deployed control uses predicted labels rather than an intervention search.

### Required files / next data

Conceptual method diagram. No trained-policy improvement or unmeasured latency number may appear.

---

## F3 — Expanded signed-credit evidence across 200 histories

**Evidence:** MEASURED aggregate data  
**Placement:** Optional replacement for main Table 1, or appendix; do not duplicate both at full size.

### Data contract / 事实锁

There are 200 controlled interference-chain prefixes and 1,000 block measurements. The metric is candidate-value credit, NOT the pilot’s mean action log-score and NOT native task return. Five row means and positive fractions are: Initial +0.000798 / 55.0%; Reference 1 +0.000103 / 50.0%; Superseded −0.007612 / 8.0%; Reference 2 −0.000135 / 50.0%; Latest +0.011020 / 98.5%. The superseded negative fraction is 92.0%. Two reference donors are used for other blocks; a reference block uses the other reference as its single donor. Only aggregate values are provided for this brief. Do not reconstruct distributions, standard deviations, scatter clouds, error bars, or AUROC from these means. A near-zero reference mean is not a claim that every reference is neutral.

### A — Diverging means and sign frequencies

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
There are 200 controlled interference-chain prefixes and 1,000 block measurements. The metric is candidate-value credit, NOT the pilot’s mean action log-score and NOT native task return. Five row means and positive fractions are: Initial +0.000798 / 55.0%; Reference 1 +0.000103 / 50.0%; Superseded −0.007612 / 8.0%; Reference 2 −0.000135 / 50.0%; Latest +0.011020 / 98.5%. The superseded negative fraction is 92.0%. Two reference donors are used for other blocks; a reference block uses the other reference as its single donor. Only aggregate values are provided for this brief. Do not reconstruct distributions, standard deviations, scatter clouds, error bars, or AUROC from these means. A near-zero reference mean is not a claim that every reference is neutral.

COMPOSITION
Use a 2.5:1 landscape figure with aligned block rows. Left 65%: a horizontal diverging dot-and-stem chart with a visible zero reference and candidate-value-effect axis. Plot exactly the five supplied means. Right 35%: a compact positive-fraction column, optionally a stacked positive/negative frequency bar using the supplied fractions. Preserve the chronological row order. Mark the two update rows with distinct plus/minus annotations and keep references muted. Include n=200 prefixes, 5 blocks each. Show no error bars unless raw paired data are later supplied.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Signed candidate-value effects in the expanded controlled study. Latest updates are positive in 98.5% of 200 prefixes, and superseded updates are negative in 92%. Reference means are near zero with balanced signs. These effects concern the fixed candidate-policy value proxy and should not be read as online success-rate differences. No uncertainty distribution is inferred from aggregate statistics.
```

### B — History-position signed profile

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
There are 200 controlled interference-chain prefixes and 1,000 block measurements. The metric is candidate-value credit, NOT the pilot’s mean action log-score and NOT native task return. Five row means and positive fractions are: Initial +0.000798 / 55.0%; Reference 1 +0.000103 / 50.0%; Superseded −0.007612 / 8.0%; Reference 2 −0.000135 / 50.0%; Latest +0.011020 / 98.5%. The superseded negative fraction is 92.0%. Two reference donors are used for other blocks; a reference block uses the other reference as its single donor. Only aggregate values are provided for this brief. Do not reconstruct distributions, standard deviations, scatter clouds, error bars, or AUROC from these means. A near-zero reference mean is not a claim that every reference is neutral.

COMPOSITION
Use a 2.6:1 canvas. The x-axis follows the five natural history positions; above it place short role labels, and below it a dot plot of mean candidate-value effect with a zero line. Use points only, or connecting dashed segments explicitly labeled “ordered role summary,” not a learned temporal evolution. In a narrow second strip, print positive fractions under the matching positions. Do not use a smooth fitted curve through five categorical roles. Let the strongly negative superseded point and positive latest point be the visual center.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Signed candidate-value effects in the expanded controlled study. Latest updates are positive in 98.5% of 200 prefixes, and superseded updates are negative in 92%. Reference means are near zero with balanced signs. These effects concern the fixed candidate-policy value proxy and should not be read as online success-rate differences. No uncertainty distribution is inferred from aggregate statistics.
```

### C — Compact evidence matrix

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
There are 200 controlled interference-chain prefixes and 1,000 block measurements. The metric is candidate-value credit, NOT the pilot’s mean action log-score and NOT native task return. Five row means and positive fractions are: Initial +0.000798 / 55.0%; Reference 1 +0.000103 / 50.0%; Superseded −0.007612 / 8.0%; Reference 2 −0.000135 / 50.0%; Latest +0.011020 / 98.5%. The superseded negative fraction is 92.0%. Two reference donors are used for other blocks; a reference block uses the other reference as its single donor. Only aggregate values are provided for this brief. Do not reconstruct distributions, standard deviations, scatter clouds, error bars, or AUROC from these means. A near-zero reference mean is not a claim that every reference is neutral.

COMPOSITION
Use a 1.9:1 figure with five memory rows and three clearly labeled numeric columns: mean candidate-value effect, positive fraction, negative fraction. Use a centered small diverging glyph for the mean but retain the exact numeric label. Put Initial and references in the same visual rank as the update rows, avoiding selective omission. This is an evidence matrix rather than a heatmap mixing arbitrary metric units. The caption explicitly states that aggregate reference cancellation remains possible and donor sensitivity needs separate analysis.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Signed candidate-value effects in the expanded controlled study. Latest updates are positive in 98.5% of 200 prefixes, and superseded updates are negative in 92%. Reference means are near zero with balanced signs. These effects concern the fixed candidate-policy value proxy and should not be read as online success-rate differences. No uncertainty distribution is inferred from aggregate statistics.
```

### Caption

Signed candidate-value effects in the expanded controlled study. Latest updates are positive in 98.5% of 200 prefixes, and superseded updates are negative in 92%. Reference means are near zero with balanced signs. These effects concern the fixed candidate-policy value proxy and should not be read as online success-rate differences. No uncertainty distribution is inferred from aggregate statistics.

### Required files / next data

Existing `tango_v2_teacher_credit200.json` aggregates and the supplied report. Raw per-prefix records are required before adding intervals or distribution plots.

---

## F4 — Real-GUI policy benefit: paired outcomes, not confidence sharpening

**Evidence:** MISSING — native evaluation required  
**Placement:** Main result figure after the corresponding experiments finish; can replace part of Table 2.

### Data contract / 事实锁

No completed real-GUI results for the recurrent SIGMA method have been supplied. Required input fields are benchmark, task_id, scenario_group, attempt_seed, training_seed, method, native_success, valid_run, error_category, inference_time, and paired identifier. Benchmarks must have separate native-success denominators. Methods: frozen shared-history base, matched action CE, successful-only CE, global-return control, action-advantage control, unsigned gate, SPD-style control, matched FocusMem if actually reproduced, and SIGMA. The main contrast is SIGMA minus matched action CE. Do not insert a positive gain, monotone baseline ordering, confidence interval, or star without the measured input. Benchmark-wide transfer cannot include training on its evaluation trajectories.

### A — Paired difference forest plot

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
No completed real-GUI results for the recurrent SIGMA method have been supplied. Required input fields are benchmark, task_id, scenario_group, attempt_seed, training_seed, method, native_success, valid_run, error_category, inference_time, and paired identifier. Benchmarks must have separate native-success denominators. Methods: frozen shared-history base, matched action CE, successful-only CE, global-return control, action-advantage control, unsigned gate, SPD-style control, matched FocusMem if actually reproduced, and SIGMA. The main contrast is SIGMA minus matched action CE. Do not insert a positive gain, monotone baseline ordering, confidence interval, or star without the measured input. Benchmark-wide transfer cannot include training on its evaluation trajectories.

COMPOSITION
Use a 2.4:1 canvas with independent panels for MemGUI-Bench, AndroidWorld, and WorkArena++. For each panel draw the measured SIGMA-minus-control task-success difference and task/scenario-clustered 95% interval on a symmetric percentage-point axis centered at zero. Directly label the paired denominator and rescue/regression counts. Use no combined cross-benchmark average unless its definition is supplied. If no results file is present, create only the panel frames labeled “Awaiting native outcomes”; do not draw marks or plausible bars.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Native GUI task success under matched observations, actions, and evaluation budgets. Differences are paired within task instance; uncertainty is clustered by task or scenario, with mirrored tasks kept together. Rescues and regressions describe changes relative to the stated control. Infrastructure failures and incomplete denominators are reported separately.
```

### B — Absolute success plus rescue/regression

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
No completed real-GUI results for the recurrent SIGMA method have been supplied. Required input fields are benchmark, task_id, scenario_group, attempt_seed, training_seed, method, native_success, valid_run, error_category, inference_time, and paired identifier. Benchmarks must have separate native-success denominators. Methods: frozen shared-history base, matched action CE, successful-only CE, global-return control, action-advantage control, unsigned gate, SPD-style control, matched FocusMem if actually reproduced, and SIGMA. The main contrast is SIGMA minus matched action CE. Do not insert a positive gain, monotone baseline ordering, confidence interval, or star without the measured input. Benchmark-wide transfer cannot include training on its evaluation trajectories.

COMPOSITION
Use a 2.8:1 canvas. Left 60%: measured native task-success dot plots for all methods, grouped by benchmark with explicit denominators. Right 40%: paired rescue and regression counts for SIGMA against action CE and unsigned gate. Use separate positive and negative sides of a zero axis, with count labels. The figure must distinguish task-level native outcomes from candidate-score metrics. Show intervals only when computed from the input records under the prescribed clustering. Missing data leave blank slots, not zeros.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Native GUI task success under matched observations, actions, and evaluation budgets. Differences are paired within task instance; uncertainty is clustered by task or scenario, with mirrored tasks kept together. Rescues and regressions describe changes relative to the stated control. Infrastructure failures and incomplete denominators are reported separately.
```

### C — Task-level heterogeneity view

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
No completed real-GUI results for the recurrent SIGMA method have been supplied. Required input fields are benchmark, task_id, scenario_group, attempt_seed, training_seed, method, native_success, valid_run, error_category, inference_time, and paired identifier. Benchmarks must have separate native-success denominators. Methods: frozen shared-history base, matched action CE, successful-only CE, global-return control, action-advantage control, unsigned gate, SPD-style control, matched FocusMem if actually reproduced, and SIGMA. The main contrast is SIGMA minus matched action CE. Do not insert a positive gain, monotone baseline ordering, confidence interval, or star without the measured input. Benchmark-wide transfer cannot include training on its evaluation trajectories.

COMPOSITION
Use a 2.6:1 canvas. The main panel ranks task groups by measured SIGMA-minus-CE success difference, preserving both regressions and improvements. A small side panel gives the total benchmark difference and interval. Each task group carries its evaluated attempt count. Do not choose only memory-heavy tasks post hoc; use the registered subgroup labels. Draw dots rather than smoothing discrete task effects. If only aggregate success counts are supplied, refuse the task-level panel and fall back to an explicitly aggregate comparison.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Native GUI task success under matched observations, actions, and evaluation budgets. Differences are paired within task instance; uncertainty is clustered by task or scenario, with mirrored tasks kept together. Rescues and regressions describe changes relative to the stated control. Infrastructure failures and incomplete denominators are reported separately.
```

### Caption

Native GUI task success under matched observations, actions, and evaluation budgets. Differences are paired within task instance; uncertainty is clustered by task or scenario, with mirrored tasks kept together. Rescues and regressions describe changes relative to the stated control. Infrastructure failures and incomplete denominators are reported separately.

### Required files / next data

Required: paired native episode records. These results are NR in the manuscript. No synthetic diagnostic accuracy can fill this figure.

---

## F5 — Which component turns credit into a useful correction?

**Evidence:** MISSING — matched ablations required  
**Placement:** Appendix or selected main ablation panel; not all variants must fit the main paper.

### Data contract / 事实锁

No final SIGMA ablation results are available. Planned contrasts are signed versus magnitude-only labels; true versus shuffled/flipped credit; action-only versus credit-only versus joint losses; GRU versus independent MLP versus small Transformer; recency-only versus measured credit; K/V/K+V; early/middle/late layers; ranks 4/8/16; no-energy versus normalized-energy training. Every contrast must share task manifests and training data. Report actual parameter counts and additional teacher-label budgets. Do not pool different metrics in one unnormalized heatmap or assert that positive credit guarantees a beneficial scaling direction.

### A — Loss and sign removal panel

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
No final SIGMA ablation results are available. Planned contrasts are signed versus magnitude-only labels; true versus shuffled/flipped credit; action-only versus credit-only versus joint losses; GRU versus independent MLP versus small Transformer; recency-only versus measured credit; K/V/K+V; early/middle/late layers; ranks 4/8/16; no-energy versus normalized-energy training. Every contrast must share task manifests and training data. Report actual parameter counts and additional teacher-label budgets. Do not pool different metrics in one unnormalized heatmap or assert that positive credit guarantees a beneficial scaling direction.

COMPOSITION
Use a compact 2.5:1 canvas containing two clearly separate panels. Panel a plots measured task-success differences relative to full SIGMA when removing signed labels, credit supervision, or policy supervision. Panel b plots normalized noncritical-policy total variation for the same variants. Both use measured paired data and labeled units. Zero is visible in the success panel. Do not label any component essential before observing the effect. Missing ablations remain “NR” without data marks.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Matched ablations separate the effect of signed supervision, controller context, and the learned correction. Performance, policy perturbation, and computational cost are reported in their own units. All variants use the same held-out task instances; missing settings are labeled rather than interpolated.
```

### B — Controller transfer and budget comparison

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
No final SIGMA ablation results are available. Planned contrasts are signed versus magnitude-only labels; true versus shuffled/flipped credit; action-only versus credit-only versus joint losses; GRU versus independent MLP versus small Transformer; recency-only versus measured credit; K/V/K+V; early/middle/late layers; ranks 4/8/16; no-energy versus normalized-energy training. Every contrast must share task manifests and training data. Report actual parameter counts and additional teacher-label budgets. Do not pool different metrics in one unnormalized heatmap or assert that positive credit guarantees a beneficial scaling direction.

COMPOSITION
Use a 2.6:1 canvas. Left: a grouped dot plot of measured in-distribution and template/history-OOD task success for MLP, GRU, and Transformer controllers. Right: measured parameter count and end-to-end latency in separate axes, or a cost-success scatter if all fields exist. Avoid promising recurrent superiority: the visual ordering follows data. Make the recency-only heuristic a separately named low-cost control. No fabricated OOD retention percentages.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Matched ablations separate the effect of signed supervision, controller context, and the learned correction. Performance, policy perturbation, and computational cost are reported in their own units. All variants use the same held-out task instances; missing settings are labeled rather than interpolated.
```

### C — Layer/target localization matrix

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
No final SIGMA ablation results are available. Planned contrasts are signed versus magnitude-only labels; true versus shuffled/flipped credit; action-only versus credit-only versus joint losses; GRU versus independent MLP versus small Transformer; recency-only versus measured credit; K/V/K+V; early/middle/late layers; ranks 4/8/16; no-energy versus normalized-energy training. Every contrast must share task manifests and training data. Report actual parameter counts and additional teacher-label budgets. Do not pool different metrics in one unnormalized heatmap or assert that positive credit guarantees a beneficial scaling direction.

COMPOSITION
Use a 2.2:1 figure. Rows are early, middle, late layer groups; columns are K, V, and K+V. Each cell may encode one and only one measured quantity: held-out native success gain over the same base. A separate narrow table provides intervention energy and confidence intervals when available. Use a symmetric zero-centered scale and retain adverse cells. Do not merge the current middle-K diagnostic means with unrun layer-group performance results. Missing cells are hatched and labeled NR.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Matched ablations separate the effect of signed supervision, controller context, and the learned correction. Performance, policy perturbation, and computational cost are reported in their own units. All variants use the same held-out task instances; missing settings are labeled rather than interpolated.
```

### Caption

Matched ablations separate the effect of signed supervision, controller context, and the learned correction. Performance, policy perturbation, and computational cost are reported in their own units. All variants use the same held-out task instances; missing settings are labeled rather than interpolated.

### Required files / next data

Required: matched ablation run records and exact model/controller parameter counts.

---

## F6 — Does local credit predict downstream return, and when does it fail?

**Evidence:** MISSING — paired return labels required  
**Placement:** Mechanism-to-application bridge, preferably appendix or one main analysis panel.

### Data contract / 事实锁

Current local action-score and candidate-value effects do not establish trajectory-return effects. Required data pair the same prefix and block under baseline and matched intervention, then obtain native returns using restored environment states and fixed continuation/intervention lifetime. Each point must retain prefix, block, donor, task group, local utility type, local effect, mean paired rollout effect, and rollout uncertainty. Sign agreement, zero effects, donor sensitivity, and failures all remain visible. Do not infer a correlation from the existing role-average table.

### A — Local-to-return scatter

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
Current local action-score and candidate-value effects do not establish trajectory-return effects. Required data pair the same prefix and block under baseline and matched intervention, then obtain native returns using restored environment states and fixed continuation/intervention lifetime. Each point must retain prefix, block, donor, task group, local utility type, local effect, mean paired rollout effect, and rollout uncertainty. Sign agreement, zero effects, donor sensitivity, and failures all remain visible. Do not infer a correlation from the existing role-average table.

COMPOSITION
Use a 2.1:1 canvas with separate plots for action-score versus native-return effect and candidate-value versus native-return effect. Plot actual prefix-block pairs, not fabricated samples around aggregate means. Include zero lines and label all four sign quadrants. Correlation and its interval appear only after task-clustered calculation. The diagonal is not a numerical equality line because units differ. Invalid-restoration cases are excluded by a prespecified rule and their count is displayed.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Testing whether an internal intervention effect predicts downstream task return. Local and native-return effects are measured on the same prefix/block under matched restoration and continuation rules. Disagreement identifies the limits of the surrogate rather than being removed from analysis.
```

### B — Signed agreement contingency

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
Current local action-score and candidate-value effects do not establish trajectory-return effects. Required data pair the same prefix and block under baseline and matched intervention, then obtain native returns using restored environment states and fixed continuation/intervention lifetime. Each point must retain prefix, block, donor, task group, local utility type, local effect, mean paired rollout effect, and rollout uncertainty. Sign agreement, zero effects, donor sensitivity, and failures all remain visible. Do not infer a correlation from the existing role-average table.

COMPOSITION
Use a 2.0:1 canvas. Main panel: a measured 3×3 contingency table of negative/neutral/positive local credit versus negative/neutral/positive native-return credit. State the separate calibrated tolerances. Side panel: paired rollout variance and donor-sign agreement, each on its own scale. Do not classify all small effects as exact zero. Include sample counts and the policy/intervention lifetime. This view is appropriate when reliable continuous correlation is unavailable.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Testing whether an internal intervention effect predicts downstream task return. Local and native-return effects are measured on the same prefix/block under matched restoration and continuation rules. Disagreement identifies the limits of the surrogate rather than being removed from analysis.
```

### C — Boundary-condition stratification

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
Current local action-score and candidate-value effects do not establish trajectory-return effects. Required data pair the same prefix and block under baseline and matched intervention, then obtain native returns using restored environment states and fixed continuation/intervention lifetime. Each point must retain prefix, block, donor, task group, local utility type, local effect, mean paired rollout effect, and rollout uncertainty. Sign agreement, zero effects, donor sensitivity, and failures all remain visible. Do not infer a correlation from the existing role-average table.

COMPOSITION
Use a 2.7:1 canvas. Show measured sign agreement or policy improvement across prespecified categories: current-value query, previous-value query, newest observation irrelevant, redundant evidence, and missing latest evidence. Each group has an interval and sample count. Do not select categories based on which favor SIGMA. A qualitative case strip may show one representative failure where local credit and final return disagree, but the screenshots must come from the corresponding run.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Testing whether an internal intervention effect predicts downstream task return. Local and native-return effects are measured on the same prefix/block under matched restoration and continuation rules. Disagreement identifies the limits of the surrogate rather than being removed from analysis.
```

### Caption

Testing whether an internal intervention effect predicts downstream task return. Local and native-return effects are measured on the same prefix/block under matched restoration and continuation rules. Disagreement identifies the limits of the surrogate rather than being removed from analysis.

### Required files / next data

Required: new native counterfactual rollout pairs. The existing 200-prefix candidate-value study is not a substitute.

---

## F7 — The cost of intervention supervision and no-hook deployment

**Evidence:** MISSING — measured timing and student outcomes required  
**Placement:** Appendix efficiency/distillation analysis.

### Data contract / 事实锁

The current method uses an expensive teacher intervention sweep during training and predicted credit at deployment. The deployed reference implementation has a clean feature prefill and an adapted policy pass; both must be charged. A no-hook LoRA student is an optional separately trained model, not an exact merge of a dynamic gate. Required records: teacher forwards, teacher GPU-hours, training GPU-hours, per-action clean-prefill/adapted-prefill/decode time, peak memory, native task success, student success, and paired confidence intervals. Gain retention is unstable when teacher gain is near zero; report absolute outcomes regardless.

### A — Cost breakdown plus absolute success

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
The current method uses an expensive teacher intervention sweep during training and predicted credit at deployment. The deployed reference implementation has a clean feature prefill and an adapted policy pass; both must be charged. A no-hook LoRA student is an optional separately trained model, not an exact merge of a dynamic gate. Required records: teacher forwards, teacher GPU-hours, training GPU-hours, per-action clean-prefill/adapted-prefill/decode time, peak memory, native task success, student success, and paired confidence intervals. Gain retention is unstable when teacher gain is near zero; report absolute outcomes regardless.

COMPOSITION
Use a 2.7:1 canvas. Left panel: measured stacked latency bars whose components sum to total time (clean feature prefill, adapted policy prefill, decoding, controller). Compare base, SIGMA teacher, and no-hook student. Right panel: native success with matched intervals for base, teacher, CE-LoRA student, and credit-informed policy-distilled student. Put training-time teacher cost in a separate annotation, not inside inference latency. No unmeasured speedup multiplier.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Compute and deployment trade-offs. Teacher-label extraction is a training cost; deployment includes every feature and policy pass. No-hook policy distillation is evaluated separately from static linear merging, with absolute task success reported alongside any gain-retention ratio.
```

### B — Performance/cost Pareto

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
The current method uses an expensive teacher intervention sweep during training and predicted credit at deployment. The deployed reference implementation has a clean feature prefill and an adapted policy pass; both must be charged. A no-hook LoRA student is an optional separately trained model, not an exact merge of a dynamic gate. Required records: teacher forwards, teacher GPU-hours, training GPU-hours, per-action clean-prefill/adapted-prefill/decode time, peak memory, native task success, student success, and paired confidence intervals. Gain retention is unstable when teacher gain is near zero; report absolute outcomes regardless.

COMPOSITION
Use a 2.3:1 canvas. Plot measured end-to-end action latency on the horizontal axis and native task success on the vertical axis. Label methods directly, with optional seed variation only from real runs. Mark a Pareto frontier only among measured points. Do not pre-draw a favorable upper-left student. A small footer lists teacher-label extraction cost. If no student data exist, retain the student as NR in a legend, not as a speculative point.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Compute and deployment trade-offs. Teacher-label extraction is a training cost; deployment includes every feature and policy pass. No-hook policy distillation is evaluated separately from static linear merging, with absolute task success reported alongside any gain-retention ratio.
```

### C — Distillation outcome ledger

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
The current method uses an expensive teacher intervention sweep during training and predicted credit at deployment. The deployed reference implementation has a clean feature prefill and an adapted policy pass; both must be charged. A no-hook LoRA student is an optional separately trained model, not an exact merge of a dynamic gate. Required records: teacher forwards, teacher GPU-hours, training GPU-hours, per-action clean-prefill/adapted-prefill/decode time, peak memory, native task success, student success, and paired confidence intervals. Gain retention is unstable when teacher gain is near zero; report absolute outcomes regardless.

COMPOSITION
Use a 2.0:1 compact evidence figure. Rows are base, dynamic gated teacher, static merged-residual baseline, ordinary CE-LoRA student, and residual-distilled student. Columns clearly separate hook usage, observable information, native success, and measured latency. The static merge belongs to the legacy linear residual and must not be shown as equivalent to removing SIGMA’s recurrent gate. Add gain retention only when a verified teacher gain and uncertainty are supplied.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Compute and deployment trade-offs. Teacher-label extraction is a training cost; deployment includes every feature and policy pass. No-hook policy distillation is evaluated separately from static linear merging, with absolute task success reported alongside any gain-retention ratio.
```

### Caption

Compute and deployment trade-offs. Teacher-label extraction is a training cost; deployment includes every feature and policy pass. No-hook policy distillation is evaluated separately from static linear merging, with absolute task success reported alongside any gain-retention ratio.

### Required files / next data

Required: native teacher/student outputs and profiler logs. No 70% retention target is a result.

---

## F8 — A real GUI case: correction, failure, and credit over time

**Evidence:** MISSING real screenshots; CONCEPTUAL layout only  
**Placement:** Appendix; one representative case can replace a main figure after evaluation.

### Data contract / 事实锁

No completed real SIGMA rollout is supplied. A valid case requires the actual chronological screenshots, instruction, executed actions, native outcome, predicted controller credits, intervention labels if collected, and matched baseline run. Predicted credit is not measured causal credit; label them separately. Select a rescue and a regression by a prespecified criterion and disclose it. Redact personal data without inventing interface evidence. A mock interface must be labeled schematic and cannot carry a claimed real success.

### A — Aligned baseline/SIGMA filmstrip

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
No completed real SIGMA rollout is supplied. A valid case requires the actual chronological screenshots, instruction, executed actions, native outcome, predicted controller credits, intervention labels if collected, and matched baseline run. Predicted credit is not measured causal credit; label them separately. Select a rescue and a regression by a prespecified criterion and disclose it. Redact personal data without inventing interface evidence. A mock interface must be labeled schematic and cannot carry a claimed real success.

COMPOSITION
Use a 2.8:1 two-row filmstrip. Rows are matched baseline and SIGMA; columns are identical task phases, with alignment maintained until divergence. Highlight the old and updated visual evidence using thin numbered outlines. Below the SIGMA row add predicted credit labels aligned to historical blocks, not arbitrary bright attention maps. Show the actual first action divergence and final native evaluator result. Use only captured screens. Without them, provide gray frame placeholders saying “real rollout required”.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Matched real-GUI examples show where memory adaptation changes an executed action. Screenshots and outcomes come from the named task instances. Predicted controller credit is distinguished from intervention-measured credit; both successful corrections and regressions are retained.
```

### B — Evidence-provenance storyboard

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
No completed real SIGMA rollout is supplied. A valid case requires the actual chronological screenshots, instruction, executed actions, native outcome, predicted controller credits, intervention labels if collected, and matched baseline run. Predicted credit is not measured causal credit; label them separately. Select a rescue and a regression by a prespecified criterion and disclose it. Redact personal data without inventing interface evidence. A mock interface must be labeled schematic and cannot carry a claimed real success.

COMPOSITION
Use a 2.5:1 canvas with one central current-decision screenshot occupying 45% of the area. To the left, two earlier screenshots show the original record and its update, connected to historical memory blocks. To the right, show the actual baseline and SIGMA action outputs and resulting screens. Solid outlines mark observed content, dashed arrows show predicted influence. Include one compact note separating measured intervention credit from predicted controller credit. No claim of causal attention localization from gate heatmaps alone.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Matched real-GUI examples show where memory adaptation changes an executed action. Screenshots and outcomes come from the named task instances. Predicted controller credit is distinguished from intervention-measured credit; both successful corrections and regressions are retained.
```

### C — Rescue and regression side by side

```text
Create a publication-ready scientific figure for “SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents.” Do not put the full paper title or a decorative method-name banner in the artwork.

SCIENTIFIC FACT LOCK
No completed real SIGMA rollout is supplied. A valid case requires the actual chronological screenshots, instruction, executed actions, native outcome, predicted controller credits, intervention labels if collected, and matched baseline run. Predicted credit is not measured causal credit; label them separately. Select a rescue and a regression by a prespecified criterion and disclose it. Redact personal data without inventing interface evidence. A mock interface must be labeled schematic and cannot carry a claimed real success.

COMPOSITION
Use a 2.6:1 canvas with two equal panels: one measured rescue and one measured regression. Each contains a minimal aligned sequence of historical evidence, decision, and outcome. Use identical visual emphasis and label selection criteria, task identifiers, and seeds. A failure panel must not be diminished or hidden. Discuss whether the failure came from credit prediction, correction direction, grounding, or execution only when logs support that diagnosis.

RENDERING AND SAFETY
Use white background, thin vector lines, simple sans-serif labels and deliberate whitespace. At 5.5-inch publication width, all text should remain readable. Positive, negative and reference roles may use muted teal, muted rust and gray, but also require plus/minus symbols or line-pattern differences. Do not add 3-D effects, brains, robots, stock interface screenshots, invented measurements, fabricated error bars, stars, or a presumed ranking of methods. Distinguish measured, derived, conceptual and missing information. Quantitative plotting must use the supplied exact data; when required data are missing, output an explicitly unpopulated layout rather than a plausible result.

FINAL CAPTION TO PRESERVE
Matched real-GUI examples show where memory adaptation changes an executed action. Screenshots and outcomes come from the named task instances. Predicted controller credit is distinguished from intervention-measured credit; both successful corrections and regressions are retained.
```

### Caption

Matched real-GUI examples show where memory adaptation changes an executed action. Screenshots and outcomes come from the named task instances. Predicted controller credit is distinguished from intervention-measured credit; both successful corrections and regressions are retained.

### Required files / next data

Required: actual baseline and SIGMA screenshot trajectories. Do not reuse controlled Interference Chain tiles as real benchmark evidence.

---

## 全图验收

1. F1 的 10-prefix action-score effect 与 F3 的 200-prefix candidate-value effect 没有放到同一个数值轴上。
2. Figure 1 的 60→100 是 **手工指定目标 span 的诊断干预**，不是训练完 SIGMA 的在线表现。
3. Positive / negative 表示原记忆相对 donor 的贡献，不表示正负 gate 一定对应正负性能变化。
4. 历史输入、current screenshot、target-action token、监督标签、controller features 的边界正确。
5. 不把静态残差 merge 画成动态 recurrent gate 的无损蒸馏。
6. 未测主结果、置信区间、延迟、返还率都保持缺失；没有为了“看上去完整”而填数字。
7. 真实 GUI 主评估在视觉层级上高于 synthetic 机制诊断；Interference Chain 不冒充应用 benchmark。
8. 图表中所有单位、数据分母、模型 revision 和实验条件可追溯到结果文件。

## 采用的风格依据

使用用户先前的 `Scientific_Figure_Style_Atlas.md`：事实锁、主结论优先、训练/推理边界、配对证据，以及每图三个不同阅读路径的方案。论文写作结构另参考 ARIS 的 paper-write 与其链接的 anti-defensive-writing-en；这些规范不授权补造结果。
