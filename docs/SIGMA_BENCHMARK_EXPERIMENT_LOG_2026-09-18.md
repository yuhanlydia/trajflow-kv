# SIGMA → AndroidWorld / MemGUI 实验交接日志

**数据截点：2026-09-18 11:56:08，Asia/Shanghai。** 本文供实验复查，记录当前服务器实际运行的 7B 权重、部分结果、故障证据与修复状态。实验仍在运行，本文是固定时间快照。

## 1. 当前结论

**截至截点，没有证据支持 SIGMA 在这两个原生 benchmark 上取得增益。** MemGUI 的 CE 对照已有 1 个成功，SIGMA 暂无成功。接入修复已经有可验证进展，但不能当作任务成功率改善。

| Benchmark / 分组 | 权重 | 已完成 / 本组计划 | 成功 / 已完成 | 当前状态 |
|---|---|---:|---:|---|
| AndroidWorld，新 `finger_swipe_v1` | SIGMA seed 11 | 5 / 329 | 0 / 5 | 第 6 个回合运行中 |
| AndroidWorld，已退役旧动作协议 | SIGMA seed 11 | 19 / 19 已开始回合 | 0 / 19 | 原始结果保留，单独记账 |
| MemGUI，当前训练后对照 | CE seed 11 | 21 / 384 | 1 / 21（4.76%） | 第 22 个任务尚未结算 |
| MemGUI，当前训练后方法 | SIGMA seed 11 | 21 / 384 | 0 / 21 | 第 22 个任务尚未结算 |

分母是**已完成回合**，不是独立任务类型总数，也不是把所有未完成回合算作失败。MemGUI 数字来自原始最终裁判记录，完整批次转换、最终可比性校验尚未完成。训练种子 23/37 已有权重，原生评测尚在当前队列后续位置；AndroidWorld 的 CE11 也尚未轮到。

机器可读附件：[固定快照](experiment_logs/2026-09-18-sigma/summary.json)、[66 条已完成回合明细](experiment_logs/2026-09-18-sigma/completed_cases.csv)、[源文件 SHA256 索引](experiment_logs/2026-09-18-sigma/source_index.json)。66 条由 Android 新协议 5 条、旧协议 19 条、MemGUI 两臂各 21 条组成，**不将它们合并成一个成功率**。

## 2. 任务范围与版本差异

本次用户要求是把已有方法扩展到 AndroidWorld 和 MemGUI，复用已完成权重，解决接入问题并完成比较。论文重建、重新寻找损失公式和其他 E01–E17 项目不是这次扩展的前置条件。

| 对象 | 实际版本 / 状态 |
|---|---|
| 服务器执行工作区 Git HEAD | `2683fb80d9e0c990d18c9f7f252661c576141cd4`，另有大量本地实现和实验文件 |
| 本次上传时抓取的 GitHub `main` | `95a625161b6678547be771dde45c2fbf5845ccf8` |
| `main` 新增协议 | Qwen3-VL-8B，500/50/50 受控数据；不是本文运行配置 |
| Android 当前数值代码快照 | `d333d8f2b9216278b2ad7d0913012b9ededd2dddc3bec3e92777c768cd9fe798` |
| MemGUI 当前数值代码快照 | `6f2608c77474c7c8f549657a91bc0c77a1654a2555957deecc06e010a0651229` |

因此，**当前服务器实验不能称为与最新 `main` 完全对齐，也不能称为 Qwen3 新协议的实验结果**。本日志发布分支基于最新 `main`；正在执行的实验继续使用其已登记源码快照。完整执行快照和大体积轨迹在服务器保存，本次附件提供配置摘要、结果明细、关键补丁和证据哈希。

## 3. 实际使用的“训练后模型”

底座为 **Qwen2.5-VL-7B-Instruct**，模型 revision 为 `cc594898137f460bfe9f0759e9844b3ce807cfb5`。当前 SIGMA/CE 都确实加载了训练后模块，不能标成 Base。

| 配置 | 当前实际值 |
|---|---|
| 底座参数 | 冻结；本轮未做 7B 底座全参数 GUI 微调 |
| 更新参数 | 控制器 856,577 + 残差模块 73,728 = 930,305 |
| 干预位置 | K，层索引 9–17；rank 8，alpha 8 |
| 控制器 | 单层 GRU，hidden 256 |
| 数据 | 每臂 200 个受控训练前缀，50 个验证前缀 |
| 数据性质 | `controlled_state_machine_not_native_GUI`；候选动作是 `select_slot_1` 至 `select_slot_8` |
| 方法标签 | CE / memory（本文称 SIGMA），各 seed 11、23、37 |
| 当前 memory 配置 | `memory_policy_objective=ce`、signed gate；energy=0、KL=0、credit=1、rank loss=0.1 |
| 训练与选取 | 配置 credit/policy epochs 各为 5；保存的验证记录 epoch 索引 0–4，六组均选中索引 1 |
| 选择依据 | offline validation 候选准确率；未用原生 TEST 成功率选权重 |

六组被选中检查点在这 50 条受控验证前缀上的候选准确率都是 100%。**这不是 AndroidWorld/MemGUI 的任务成功率。** 受控槽位选择与真实截图下自由生成点击、输入和多步计划之间存在迁移差距。现有证据尚不能量化它贡献了多少失败。

当前两个检查点：

- SIGMA11：`outputs/sigma_controlled_7b_20260917/memory_s11/best.pt`，SHA256 `d9b5f71547c5bf6066a66e119d9ff16edb96c667a6d0b51092f0eae6eccf3fc4`。
- CE11：`outputs/sigma_controlled_7b_20260917/ce_s11/best.pt`，SHA256 `7f1d4ecfb6ed3ae152368a35700e2942555d15280ac6a4a3ebd0427d04734ba0`。

六组完整配置、权重摘要及各 epoch 的验证指标见 [training_summary.json](experiment_logs/2026-09-18-sigma/training_summary.json)。这些事实用于核对本次到底跑了哪个已有检查点，不额外要求重建论文训练方案。

## 4. 评测设置与可比范围

| 项目 | AndroidWorld | MemGUI |
|---|---|---|
| 完整单臂计划 | 116 个任务类型 × 3 次，共 348 回合 | 128 个任务 × 3 次，共 384 回合 |
| 重复编号 | 101 / 202 / 303；环境 suite seed 42 | 101 / 202 / 303；每次独立 pass@1 |
| 当前 SIGMA11 计划 | 旧协议 19 条保留；新协议仅继续未开始 329 条 | 当前第一批 128 个任务进行中 |
| 历史画面上限 | 5 | 5 |
| 图像像素预算 | 当前 401,408；历史 100,352 | 同左 |
| 实际生成 token 上限 | 启动命令 `--max-tokens 8192` | 环境 `TANGO_MAX_TOKENS=8192` |
| 步数预算 | 固定 30 步 | 原生 runner 的逐任务预算，例如任务 021 为 91 步 |
| 结果判定 | 原生环境任务分数 | 当前登记的外部裁判：`gpt-4.1-mini-2025-04-14` |
| 物理 GPU | 1 | CE 用 2，SIGMA 用 3，独立模拟器 |

MemGUI 明确登记为 alternate-judge 协议，本文不声称与论文默认裁判设置完全等价，也不将三次 pass@1 汇总偷换成 pass@3。

修复带来的分组限制：

1. AndroidWorld 新旧动作协议分开统计。新协议 SIGMA11 与 CE11 的共同案例为 329；再与已登记 Base 剩余回合配对时为 244。Base 已在现有队列末尾登记，不另建队列。
2. MemGUI 两臂的第 001–021 个任务使用旧快照时钟；第 022 个任务起已验证新的 `snapshot_host_clock_v1` 初始化。最终比较须同时匹配案例和有效环境协议。
3. 原 MemGUI 汇总器在完整 128 任务批次结束后才转换，不能把中间的 `completed_cases=0` 理解为没有执行任务。

**本次整理额外发现的记录问题：** Android 计划顶层 `max_new_tokens` 和一份比较摘要仍写 160，但已登记 argv 及实际启动使用 8192，应以有效启动/运行 provenance 核验预算。这是记录一致性缺口，尚未改动冻结计划。旧 Android 终结回合索引有 18 行，而终结 `episode.json` 与交接清单有 19 条；原因是第 19 条终结文件写完后在回合边界交接，汇总索引尚未追加。附件按 19 份原始终结记录计数，没有重跑。

详见 [实际协议摘要](experiment_logs/2026-09-18-sigma/protocols.json) 和 [修复/比较审计](experiment_logs/2026-09-18-sigma/repair_evidence.json)。

## 5. 失败原因：已有证据与仍未证明的部分

### 5.1 基本动作选择和任务推进很弱

Android 新协议完成的 5 个回合共 150 步，**150 步均为 `answer`，没有点击、输入或打开目标应用，非法动作数为 0**。

- `CameraTakeVideo` 的两个回合各重复 30 次 `{"action_type":"answer","text":"The task is completed."}`，但没有实际录像。
- `ClockStopWatchPausedVerify` 的三个回合反复回答画面中没有秒表，没有进入应用并完成暂停。
- 较早旧协议的 12 回合审计为 360 步：331 次左滑、29 次右滑、0 次点击，非法动作为 0。

这些属于合法动作下的任务失败，不能全部解释为 JSON 格式问题。修复水平滑动方向也无法自动解决反复声称完成的问题。详见 [新协议逐回合动作摘要](experiment_logs/2026-09-18-sigma/android_completed_action_summary.json)。

实现审计还发现，当前历史 KV 干预作用于历史图像区域；首步没有历史图像。因此首步就选错动作，不能直接归因为历史记忆取舍，也不能据此排除基础策略、提示或生成接入问题。

### 5.2 MemGUI 既有真实操作成功，也有模型和应用失败

CE 的唯一成功是 **015-CreateSimpleNote**：原始裁判判定成功，创建 Joplin 笔记 `Shopping List`，内容 `Milk and bread`。这证明该流程存在可成功的执行路径，不证明 SIGMA 有增益。

第 019/020 个购物任务出现空白分类/商品区。CE 第 019 个任务的[原始截图](experiment_logs/2026-09-18-sigma/meesho_network_error.png)明确显示联网错误。另一些失败轨迹则反复操作错误应用或错误位置；例如 SIGMA 第 020 个任务被裁判记录为反复打开 Google Assistant，而非完成 Meesho 商品比较。

原始裁判文字见 [memgui_selected_judgments.json](experiment_logs/2026-09-18-sigma/memgui_selected_judgments.json)。裁判解释是证据的一部分，不能仅凭一句评分理由就证明底层根因，也不能把所有零分都归为模型能力不足。

### 5.3 接口/后端错误

- 已确认并修复：动作 JSON 包装兼容、模型像素坐标到原图坐标映射、应用别名/启动映射、Android 水平 swipe 方向语义。
- 新确认：`input_text` 的 `text=""` 进入后端“跳过输入”分支后，变量 `ret` 没有赋值，返回 HTTP 500。已准备返回明确 no-op 的[小补丁](experiment_logs/2026-09-18-sigma/repairs/empty_input_text.patch)，5 项针对真实 `step` 函数的测试通过，**尚未部署进当前冻结后端**。

空文本补丁只消除后端异常；模型持续生成空输入仍然无法推进任务。部署该补丁需要在已完成边界登记后续运行身份，不能直接改动活跃批次的冻结后端文件。

### 5.4 环境与评测服务

- 模拟器恢复旧快照后时间回退：SIGMA 落后约 34.5 小时，CE 落后约 2.9 小时，虽然自动时间已开启。SIGMA 系统日志存在证书/OCSP 有效期检查错误。
- 两台模拟器都能解析并 ping 通 Meesho 域名，因此不是简单的“整个模拟器断网”。时钟、外部服务可用性和其他网络问题仍需分别判断。
- 校时补丁已在两臂第 022 个任务初始化前真实生效：SIGMA 误差从 -124,283.38 秒降到 -0.52 秒，CE 从 -10,433.92 秒降到 -0.99 秒。截点时这两个修复后回合尚未最终结算，不能声称修复提高了成功率。
- 裁判存在连接重试和 TPM429 限流。截点日志匹配数为 SIGMA 56 行连接错误 / 198 行限流、CE 140 / 181；**它们是日志行数，不是独立失败请求或任务数**。
- 已知重试配置缺口仍在：外部配置 `MEMGUI_MAX_RETRIES=1`，评测路径中仍有默认 200 次重试的实现。活进程未因观察超时被重启。

## 6. 已做修复及验证状态

| 修复 | 验证证据 | 部署状态 |
|---|---|---|
| JSON 包装、Qwen2.5 坐标和应用启动兼容 | 保留历史原始动作和后端动作；独立 CPU/接口检查 | 当前队列已使用 |
| Android 水平 swipe | 126 项回归检查；实际后端函数 8 个方向/动作组合检查通过 | 11:09 后新协议队列已运行；旧 19 回合保留 |
| 快照初始化后校时 | 8 项机械测试；两台设备真实时间读回；零回合重放 | 11:29 安装；CE 11:36、SIGMA 11:39 在任务 022 生效 |
| 空输入导致 HTTP500 | 原函数错误复现，修补后空输入、普通文本、Unicode、空格及真实 ADB 失败路径共 5 项检查 | 已准备，未部署 |

校时通过现有同步初始化所调用的授权辅助脚本执行，不手动修改运行中回合的时钟。该辅助脚本不在旧运行身份探针的覆盖范围，所以另有带哈希的环境修订和逐回合校时回执；最终比较必须纳入它，不能只看旧 `protocol_id`。

相关小型代码附件：[手势转换](experiment_logs/2026-09-18-sigma/repairs/android_gestures.py)、[校时模块](experiment_logs/2026-09-18-sigma/repairs/snapshot_clock.py)。这些测试验证软件行为，**不计为 benchmark 成功回合**。

## 7. 速度与资源

当前只使用物理 GPU1/2/3，GPU0 不参与。MemGUI 两臂已用两个独立模拟器并行；AndroidWorld 顺序使用独立环境。模型驻留显存不等于 GPU 持续满算力。

Android 新协议首个录像回合耗时 497.2 秒，其中记录的模型生成总时间为 36.0 秒。大部分时间消耗在交互、截图、等待和执行链路。MemGUI 还受外部裁判限流影响。因此只增加推理算力，不能按卡数同比缩短总耗时；当前不提供未经验证的全量完成时间承诺。

## 8. 建议师兄优先核对的三件事

1. **检查点是否是预期迁移的原方法权重。** 当前实际是 Qwen2.5-VL-7B 冻结底座 + 200 前缀受控训练模块；远端新 Qwen3/500 协议属于另一配置。应先核对方法、训练数据和权重绑定，避免用错实验对象。
2. **在独立开发样例上检查完整策略调用。** 先验证基础模型能否完成最小打开应用、点击、输入与导航，再固定条件比较残差开/关；当前正式 TEST 不用于修改提示、超参数或筛选重跑。
3. **统一有效评测协议与故障归因。** 补齐 token 预算记录、时钟修订、空输入后端补丁和裁判重试配置；按匹配案例/协议做最终比较，并保留基础设施失败与缺失结果。

当前数据无法证明“SIGMA 已有效”，也不足以给出“方法本身无效”的因果结论。基础策略较弱、真实环境/接口故障、受控训练到原生 GUI 的迁移差距同时存在。

## 9. 服务器复查入口

服务器工作目录：`/data/cwj/trajflow-kv-longgoal`。下列路径均相对该目录；GitHub 附件中的哈希可与原始文件核对。

| 对象 | 路径 |
|---|---|
| 本文固定快照 | `results/sigma/S00/hourly_supervision/20260918T115608+0800.json` |
| 当前运行与队列指针 | `docs/.loop_state.json` 中 `trained_native_7b.coordinators`、各 `current_child.json` |
| Android 新协议原始轨迹 | `results/sigma/E06/memory_s11_qwen7b_gesture_v1/episodes/` |
| Android 旧 19 回合 | `results/sigma/E06/memory_s11_qwen7b_compat_v2/episodes/` |
| MemGUI SIGMA | `results/sigma/E05/memory_s11_qwen7b_compat_v3/attempt_101/official_logs/` |
| MemGUI CE | `results/sigma/E05/ce_s11_qwen7b_gpu2_parallel_v1/attempt_101/official_logs/` |
| 手势修复与案例分区 | `results/sigma/S00/android_swipe_repair_v1/` |
| 校时修订与逐案例回执 | `results/sigma/S00/memgui_snapshot_clock_v1/` |
| 待部署空输入补丁 | `results/sigma/S00/memgui_empty_input_v1/` |
| 训练权重与选择记录 | `outputs/sigma_controlled_7b_20260917/` |

本文整理期间未中断已有回合、未启动额外实验队列，小时监督仍开启。旧 3B、旧 Android 和原 MemGUI 调度器均不得恢复；查看活跃权重时以当前协调器指针为准。
