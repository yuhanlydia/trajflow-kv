# SIGMA — ICLR 2027 未完成实验总计划

论文：**SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents**
目标仓库：`yuhanlydia/trajflow-kv`，默认分支 `main`。
同步日期：2026-09-15。来源：本轮 `SIGMA_ICLR2027_Rewritten.zip` 中唯一的 `main.tex`、`FIGURE_PROMPTS.md`、`EVIDENCE_AND_REVISIONS.md`，以及此前交付的实验代码包。

## 0. 给执行 agent 的第一条指令

执行下列未完成实验，不再扩展研究方向。真实 GUI 的 native task success 是主评估；Interference Chain 是内部因果分析。保留所有正、负、无差异结果，不为达到预期排序而改测试集。本文的实验 ID、依赖顺序、记录格式和产物路径是本次执行管理补充，不是新的实测结果。

**状态必须分开：** `NOT_RUN`、`BLOCKED_CODE`、`BLOCKED_DATA`、`BLOCKED_ENV`、`RUNNING`、`COMPLETE`、`FAILED`、`DEFERRED_TEACHER_GATE`。COMPLETE 只表示按协议完成，不表示方法获胜。所有下列性能实验初始均为 NOT_RUN；发现依赖缺失后写明具体阻塞原因。

已有观察不要重写成新方法结果：10-prefix matched-K pilot、200-prefix/1000-block 因果标签、旧 MiniWoB 330/400 vs CE 333/400、旧 60/60 memory trainer、Bluetooth 0/4→4/4、静态 residual merge，均有各自历史协议。它们不等于 recurrent SIGMA 已在真实 GUI 上胜过 baseline。

## 1. S00 — 代码交付与稿件配置对齐（先做）

**仓库迁移状态：** 本次写入的目标是用户指定的 `yuhanlydia/trajflow-kv` / `main`。写入前仓库为空；本次提交仅加入实验计划、机器可读清单与 README，不迁入原始代码或此前 overlay。原始代码来源为 `Yunbo-max/trajflow-kv`；执行 agent 须先完成代码迁入与下列接口核对，再把相应阶段从 BLOCKED_CODE 改为可运行。不能把旧仓库的只读权限失败记成当前仓库写入失败。

此前 `TANGO_ICLR2027_code_overlay.zip` / `TANGO_ICLR2027.patch` 是交付包，不应假定已在远端。先检查这些路径：

- `tango_iclr/`、`tests_iclr/`、`configs/tango_iclr_release.yaml`
- `scripts/run_tango_iclr.sh`、`scripts/run_tango_memgui.sh`
- 原有 `scripts/estimate_kv_memory_credit_qwen.py`、`trajflow_kv/projector.py`

缺少交付包时记 BLOCKED_CODE，由有写权限的 agent 应用已交付补丁；不要覆盖其他 agent 的未提交工作。不要把旧代码中的 TANGO 路径当作第二篇论文，路径兼容保留，报告方法名统一 SIGMA。

**新稿与旧包默认值有实际差异，必须先对齐：**

| 项目 | 单版本 SIGMA 稿件协议 | 旧 overlay 默认值，仅作迁移核对 |
|---|---|---|
| Backbone | Qwen2.5-VL-3B-Instruct，固定 revision | 同系列，revision 可能为空 |
| Scale model | Qwen2.5-VL-7B-Instruct | 不自动替换成 Qwen3 |
| Controller | width 256，单层 GRU | hidden 128 |
| Controller/joint epochs | 5 / 5 | 20 / 3 |
| Controller/residual LR | 3e-4 / 1e-4 | 1e-3 / 1e-4 |
| Accumulation | 8，batch 1 prefix | 4 |
| Training seeds | 11, 23, 37 | 0, 1, 2 |
| Evaluation attempts | 101, 202, 303，独立 reset | 必须检查 runner 是否支持 |
| History | 5 past + current；另测 8/12 past | history_limit 8 |
| Historical/current pixel caps | 100352 / 401408 | 单一 max_pixels 100352 |
| KV residual | middle K，layers 12–23，rank 8，alpha/r=8/8 | 接近，但逐项核实 |
| Energy | normalized energy；lambda_E=0/10/100/1000/3000 验证搜索 | 不能直接沿用旧 0.01 的语义 |
| Credit | lambda_c=1，lambda_s=.2，lambda_r=.1；Tg=.5/1/2 | 需核查 sign/rank loss 和温度支持 |

不要只改配置键而忽略实现。两种图像上限、history 定义、GRU 层数、梯度累积、loss normalization、checkpoint selection、三次 reset 若当前代码不支持，先补实现与测试并记录差异。若为资源改协议，所有 matched arms 同步修改，先锁定修改，再更新稿件；不能静默降低某一种方法的输入预算。

输出：`results/sigma/S00/implementation_audit.json`、`config_resolved.json`、代码 SHA、CPU 测试和真实 Qwen smoke 日志。核实旧 runner 是否确实覆盖下列所有项目；旧版“302 jobs”不等于本计划全部已实现。

## 2. 全局实验合同

### 2.1 数学和数据

分开保存三种 credit：

1. `local_score`：正确动作平均 token log-score 的原始减 matched-patch 差值。
2. `candidate_value_proxy`：softmax(candidate scores / T) 与固定 candidate Q 的内积差。不是 native expected return。
3. `native_rollout_return`：从同一可恢复环境状态出发的配对后续 rollout 最终 return 差。

不同单位不混合归一化、不画在同一数值轴。符号相同不意味着幅度可比。正/负 finite patch effect 也不自动等于正/负 K scaling 的有益方向；credit predictor 和 policy residual 必须分别受监督。

真实 baseline 使用相同 instruction、历史截图、历史动作、当前图、动作空间、parser、步数与像素预算。去掉 task-specific 硬编码正确坐标。candidate ranking 与自由生成分 track；无 gold action/candidate Q/semantic role 进入部署输入。额外 teacher labels 与调用预算单独计费，并给出同数据和同总预算两个比较口径。

### 2.2 切分与选择

固定 train/validation/test manifests、任务组、镜像对、nonce、截图哈希和采样种子后再跑。所有同实例 prefixes/candidates/donors 同组，不能按 candidate 行随机切分。测试上不筛难度、不调参数、不选 donor、不挑成功案例。最终 checkpoint 用独立 validation native task success 选择；若仅有受控 validation，明确它是不同选择规则，不能写成真实 GUI validation success。

### 2.3 运行和统计

单 GPU 单模型进程串行；记录 BF16/量化、依赖、真实显存、OOM 与重试。环境端可以是独立 KVM 主机，不因模型 GPU 机器没有本地 KVM 就伪造环境结果。设置变动导致的重跑保留旧记录。

每个 learned arm 使用 training seeds 11/23/37，每个测试实例 evaluation seeds 101/202/303。冻结 Base 不必重复“训练”三次；可复用同一合法 Base 配对评估，但不可把复制行当独立样本。环境参数 seed 和 decoding seed 是不同字段。

按任务/场景聚类 bootstrap，MemGUI mirrored tasks 同组；报告 training-seed variation。报告完整计划分母、完成覆盖率、native success、paired delta、rescues/regressions、95% CI；重复任务下 McNemar 为补充。基础设施失败单列并保留不完整运行界限，不偷偷丢弃或记为成功。不得把平均三次 pass@1 写成 pass@3。

## 3. 逐项未跑实验

### E01 — 数据冻结与外部迁移切分

- 受控集：200 critical prefixes/Template A 训练，50/Template B 验证，200/Template C untouched test；每实例独立随机 nonce。
- 已有200-prefix teacher extraction 不充当这次独立 test。
- 外部训练来源使用 disjoint mobile trajectories/AITW demonstrations；AITW action demos 不自动提供真实失败 return 或反事实 Q。
- 全套迁移：MemGUI 128 tasks、AndroidWorld 116 task definitions 全部禁止进入训练与选择；公开 evaluation trajectories 同样禁止。
- 如需要同域训练，另命名 in-domain track 并做 group-disjoint split，不再报告 full-suite unseen transfer。
- WorkArena++ 200 workflows 在看结果前固定；截图与 accessibility-assisted 两个 track 分开。
- 产物：`data/sigma/manifests/*.jsonl`、`split_audit.json`、`dataset_fingerprints.json`。

### E02 — 部署与 GPU 接口 smoke（不是重复因果发现）

- 验证在线模型真的接收多张历史图，历史-当前顺序与离线一致；所有 candidate 共用 target-free prefix features。
- candidate 顺序打乱不得改变 controller features 或预测 credit；禁止 target-action token 泄漏。
- 测 padding/missing blocks、self-patch identity、equal-span donor、pre/post-RoPE 位置约定、clean donor 与 same-forward donor 的标识。
- 仅选 decoder K/V，确认没有误 hook vision encoder；frozen backbone 无参数更新，controller/residual 有非零梯度，禁用/零 residual 恢复 base。
- 对同一 prefix 比较训练评分、重新 prefill 和 generation/cache 路径。把实际 generation 中 gate 生效写入 trace，避免离线有 gate 而在线无 gate。
- Native smoke：MemGUI/AndroidWorld 各一个安全任务；WorkArena++ 一个服务实例任务。初始化、固定合法动作、观测、native score、tear-down 全部健康。
- 产物：`results/sigma/E02/{cpu,gpu,native}_smoke.*`。schema/梯度失败记 FAILED；缺权重/环境记 BLOCKED，不扩大训练。

### E03 — 正式 teacher labels：可迁移的 clean donor 信号

- 冻结 Qwen2.5-VL-3B，默认 middle K 12–23；先10-prefix完整 extraction，接口正确再跑E01训练集。
- 一个 block 至少2个不同合法 donor（可用时）；保存每 donor 的原始 effect。reference 自替换不是多 donor。
- 比较 local_score 与 candidate_value_proxy，保留 legacy same-forward 与 clean-prefill 两种实验标签，不追溯改名。
- 跨实例 donor 不含目标答案；长度不等直接记录不可匹配，不静默插值。
- 单 donor CI 不作不确定性证据；双 donor bootstrap 只解释为描述性 donor 敏感度。
- 训练归一化统计只从训练集拟合；mask 缺失/不确定标签，不把它们当作 zero credit。
- 产物：`results/sigma/E03/teacher_{utility}_{target}_{layers}.jsonl`、donor明细、耗时和输入hash。

### E04 — 冻结骨干的 SIGMA 正式训练与主对照

按S00新稿默认先训练controller5 epochs，再policy+credit联合5 epochs；验证选择checkpoint。SGD/Adam实现以稿件AdamW为准，不换新方法。

主表 arms：Base、matched action CE、successful-only CE、Global Return、reference-weighted Action Advantage、unsigned trust gate、SPD-style GUI control、可对齐的完整 FocusMem、SIGMA。

关键 CE 对照保持同样 controller+residual，仅去掉 credit监督；不是只训练更弱的头。Global Return只能用真实logged return；Action Advantage记录Q来源和reference weights，不把无权重枚举当精确policy gradient。SPD-style是简化对照，不能写“完整SPD复现”。unsigned gate也不是FocusMem官方实现。

同一验证预算、同一训练split、3 training seeds；先完成Base/CE/SIGMA的真实部署链，再填主表其他arms，不用新增toy成绩取代主表。

产物：`outputs/sigma/{arm}/seed_{seed}/`下保存checkpoint、resolved config、训练曲线、可训练参数数、best-checkpoint规则与选择记录。

### E05 — MemGUI-Bench真实主性能表

- 128 untouched tasks；每个 learned checkpoint 独立reset评估3次。每个learned arm为128×3 training seeds×3 attempts=1152 method-episodes。
- Base每实例3次，不虚增训练种子样本。全部arms使用同一task与attempt manifest。
- 保留官方runtime和evaluator；额外评估API不与policy endpoint混用；记录官方指标定义与评估模型版本。
- 主指标native task success/pass@1平均；native pass@3另跑或按相应完整协议聚合。补充官方IRR/MTPR/FRR（仅在原生定义与记录可用时），steps、invalid action、覆盖率。
- 主比较SIGMA vs matched CE；其次vs Action Advantage、Global Return、unsigned gate；报告差值CI和task-level wins/losses。
- 产物：`results/sigma/E05/{arm}/seed_{s}/attempt_{a}/episodes.jsonl`、native logs、`paired_summary.json`。
- 对应：正文Table `tab:real`，Figure F4。缺KVM/runtime/evaluator/API则BLOCKED_ENV，不切换toy冒充。

### E06 — AndroidWorld真实移动端迁移

- 116 untouched task definitions；固定环境参数实例、独立decode attempts、同样3 training seeds协议。
- 每个learned arm 116×3×3=1044 method-episodes（每任务一个预固定参数实例；额外参数实例单独扩充分母）。
- 使用native state evaluator；HTTP task_idx不是decode seed。不能把之前Bluetooth候选坐标带入free-action主表。
- 环境健康/固定合法脚本/恢复失败单列；无可靠模拟器不继续大规模记分。
- 报告native success、rescues/regressions、per-task delta、steps、invalid action与所有infra失败。
- 产物：`results/sigma/E06/...`；正文 `tab:real`，Figure F4/F8。小规模安全smoke不替代116任务主表。

### E07 — WorkArena++真实浏览器迁移

- 200预固定workflows，真实可用服务实例；不同任务的初始化与清理独立执行。
- Screenshot-only作为视觉track；accessibility-assisted若跑则独立报告，不与之混分。
- 同样训练/评估种子与arms；每learned arm 200×3×3=1800 method-episodes。
- 使用native success与相同BrowserGym action接口；不把BrowserGym四类MiniWoB算作WorkArena++。
- 产物：`results/sigma/E07/...`；正文 `tab:real`，Figure F4/F8。无服务账号记BLOCKED_ENV。

### E08 — Credit符号、额外容量和简单记忆策略对照

固定模型、split、layers、rank、预算：signed credit / magnitude-only / shuffled / sign-flipped / unsigned；再加ungated residual、recency-only、designer-label oracle。

shuffled/sign-flipped仅改训练标签，评价仍用独立真实标签；保存具体置换种子。不要假定latest必为正或stale必为负。designer oracle单列非部署对照。

主指标native success，其次sign macro-F1、harmful detection、neutral false-positive、非关键状态TV。结果可以相同或反向，不预填单调排序。

产物：`results/sigma/E08/sign_specificity.csv`及配对日志；正文 `tab:ablation`，Figure F5。

### E09 — Controller结构和损失消融

- Independent MLP vs single-layer GRU vs two-layer small Transformer；调整宽度使容量可比，同时报告实际参数数与延迟。
- Credit-only、policy-only（matched CE）、joint；去sign loss、去ranking loss、去credit auxiliary。
- 固定预测controller、关闭residual；检查“会预测credit”是否等于“会纠正动作”。
- 主指标native success、OOD success、cost-success；不以训练loss更小取代性能。
- 产物：`results/sigma/E09/controller_loss.csv`、learning curves、3seed结果；Figure F5/F7。

### E10 — KV目标、层与rank

- K-only/V-only/K+V；early/middle/late thirds；rank4/8/16。
- 默认3B middle K rank8，单轴变化先于任意交叉组合；7B层分组按实际模型config，不能硬套36层。
- 目标/层改变时重新生成相应teacher labels，或明确命名“固定teacher跨位置迁移”独立对照。
- 报告native success、credit质量、energy、参数数、显存与延迟；仅单位置有效也是有效实验结果。
- 产物：`results/sigma/E10/kv_layer_rank.csv`；Figure F5，不用制造不存在的完整网格。

### E11 — Donor稳健性、绝对中性与符号泛化

- Single vs multiple clean donors；same-forward vs clean-prefill；equal-shape随机匹配donor对照；self-patch identity。
- 对每prefix/block保存mean、absolute effect、sign agreement、donor dispersion，而不只保存role均值。
- Neutral tolerance在训练/验证固定；report per-instance neutrality，不能从均值≈0或正负各半推出每实例无影响。
- 对独立Template C和外部历史重复；训练时选定donor规则后不在test选最有利donor。
- 主指标credit sign一致性、macro-F1/MAE、native policy与donor选择敏感度。
- 产物：`results/sigma/E11/donor_effects.jsonl`、`neutrality_summary.json`；Figure F3/F5/F6。

### E12 — 非recency捷径、长历史与OOD

- E01的Template C untouched test；新nonce、layout、update count、distractor count；历史5/8/12 past screens。
- 包含latest observation无关、较旧记录仍权威、查询previous value的retrospective任务。任务有效性由任务定义，不由预想gate符号决定。
- 不更改所有arm共同的可见信息；截断规则固定，不用gold选择保留哪些图。
- 主指标critical accuracy、native task success、sign macro-F1和harmful detection；分别报告ID与各OOD轴，不把多种OOD混成一个均值。
- 产物：`results/sigma/E12/ood_summary.csv`、逐prefix与逐task记录；正文 `sec:ablation`，Figure F5/F6。

### E13 — 从局部score到真实future return的验证

- 从disjoint training/dev原生任务选可恢复prefix；干预仅在当前决策生效，之后按固定continuation policy继续。持久干预另命名。
- baseline与patch恢复同一环境状态，使用paired随机流；固定D个donor和K次continuations，K由development的成本/方差检查确定后锁定。
- 计算 `mean_{d,k}[R(base)-R(patch_jd)]`，保留每对return、state哈希、actions和终态评分。
- 在同一prefix/block比较local_score、candidate_value_proxy、native_rollout_return：相关性、sign agreement、置信区间和不一致样例。不得互换命名。
- 恢复不一致/ANR/API失败不作为负reward训练标签。实验组不能读取测试gold给部署gate。
- 产物：`results/sigma/E13/paired_rollouts.jsonl`、`local_vs_return.csv`；Figure F6。无runtime只完成proxy部分，native项保持BLOCKED。

### E14 — Energy、保留性、效率和失败分析

- lambda_E=0/10/100/1000/3000使用稿件normalized energy；固定最终validation操作点，不照搬旧loss尺度的97.8%“保留”说法。
- 非关键策略扰动用normalized candidate policies的total variation；另测disjoint AITW action log-score等保留性，注明它不是online成功率。
- 记录teacher donor抽取成本、controller训练、clean feature prefill、adapted prefill、decode、GPU-hours、p50/p95 latency、峰值显存与可训练参数。
- 相同输入预算比较成本，Base不强加无用extra prefill；量化/图像上限变动另命名。
- 真实rollout按预注册类别标注stale interference、retrieval、grounding、parser、execution、infra；同时展示rescue/regression/unchanged案例。
- 产物：`results/sigma/E14/efficiency.csv`、`retention.csv`、`case_index.json`和脱敏截图；Figure F7/F8。

### E15 — 同模型no-hook policy distillation（有条件后做）

- 先在独立validation原生任务确认teacher存在可测增益；若没有，记DEFERRED_TEACHER_GATE，不假装已完成student贡献。
- 同架构LoRA student，无teacher hook；teacher与student得到同一可见历史。对照：teacher action CE、policy-only LoRA、KL distillation、centered residual distillation；静态merge只作静态对照。
- KL权重非负；centered logits去除共同偏移。缺信息的student不能被要求复现privileged teacher。
- 报告teacher/student/base绝对native success、paired delta、增益保留比及不确定性；teacher增益接近0时比值不稳定，报告绝对值不强算“70%”。
- 每student重新跑no-hook在线评估与保留性；静态merge结果不能代表recurrent controller已被内化。
- 产物：`results/sigma/E15/student_{objective}/...`；正文 `sec:distillation`，Figure F7。

### E16 — 模型规模复制

- Qwen2.5-VL-7B-Instruct；主对照Base/matched CE/SIGMA。保持输入、action、任务manifest与训练来源一致。
- 重新检查层数、heads、K/V宽度、vision spans与精度；重新生成该backbone的teacher labels，不直接跨模型复用raw K/V或3B labels作默认。
- 同一原生主benchmark复制；完整7B消融不是第一轮依赖。新硬件配置/量化记录完整。
- 产物：`results/sigma/E16/scale_replication.csv`、3训练seed与native日志；Figure F4/F7。

### E17 — 主表、消融、图与提交前结果闭环

- 由完整manifest生成 `tab:real`、`tab:ablation` 和对应F3–F8数据；F1/F2已有图不要改成新方法获胜图。
- 导出每方法每task每training/evaluation seed完整行，计算task/scenario clustered 95% CI；报告缺失pair与coverage。
- 同数据预算和额外teacher预算一起报告；FocusMem完整实现若未对齐仍标NR/BLOCKED，不用unsigned gate顶替名字。
- 每个结果记录 source commit、model revision、checkpoint、data与config hash、native evaluator版本、完成状态；paper单元格通过source路径回溯。
- 未运行保留NR；不生成虚构误差棒、星号、热图值、性能曲线或“已胜过CE”叙述。
- 产物：`results/sigma/report/experiment_ledger.json`、`main_real_gui.csv`、`ablations.csv`、`paper_results.tex`、`figure_manifest.json`、`failures.md`。

## 4. 执行次序与依赖

1. S00→E01→E02：代码、split和真实部署链。
2. E03→E04：正式teacher与matched训练。
3. 优先E05；E06/E07的环境依赖独立处理。mobile环境受阻时可以运行已健康browser环境，但不能把浏览器成绩代填AndroidWorld。
4. E08–E12、E14在同一冻结协议上补对照/消融；不要先扫全部组合再选最有利主结果。
5. E13验证credit与native return关系；E16复制到7B。
6. E15有teacher validation增益后再跑；E17汇总所有完成和未完成项。

性能主结论以预注册primary contrast SIGMA−matched CE的效应量与task-cluster CI为准，其他对照一起披露。不能保证录用，也不能规定数据必须出现“SIGMA > ActionAdv > CE”。达到或未达到门槛都如实记录。

## 5. 可复用入口与尚需接线的边界

以下命令仅在S00确认此前overlay已安装、且新稿配置已经对齐后适用：

```bash
python -m pytest -q tests_iclr
python -m tango_iclr.preflight --output outputs/sigma_preflight.json
bash scripts/run_tango_iclr.sh --profile all
bash scripts/run_tango_iclr.sh --profile smoke --execute
bash scripts/run_tango_iclr.sh --profile core --execute --continue-on-error
bash scripts/run_tango_iclr.sh --profile ablations --execute --continue-on-error
# E15的teacher gate满足以后：
bash scripts/run_tango_iclr.sh --profile distill --execute --continue-on-error
```

不要假设 `--profile all` 已包含本文所有消融。对每个实验ID在implementation_audit.json填写真实入口与command argv；缺recency/magnitude-only/no-residual/独立current-pixel cap等支持时先补代码，不把计划命令当成已实现。

MemGUI保持原生 `scripts/run_tango_memgui.sh` / official runtime 逐checkpoint执行；native branch-credit使用 `python -m tango_iclr.rollout_credit --help` 的实际schema；AndroidWorld/BrowserGym使用已核实的registry与manifest。不要编造endpoint、任务ID、训练结果或脚本参数。

## 6. 统一结果记录最少字段

`experiment_id, arm, implementation_status, execution_status, code_sha, model_revision, checkpoint_sha, train_seed, eval_seed, environment_instance_id, task_id, scenario_group, split, config_sha, data_sha, credit_kind, intervention_lifetime, donor_ids, native_success, invalid_action_count, infra_status, runtime_seconds, peak_memory, artifacts`。

执行agent每完成一个stage更新ledger、提交config与compact结果和日志索引；不得把截图中的私密资料、API keys或大模型权重提交GitHub。结束时明确列出COMPLETE/FAILED/BLOCKED/DEFERRED与未跑原因。
