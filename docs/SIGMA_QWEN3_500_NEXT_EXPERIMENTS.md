# SIGMA 下一轮：Qwen3-VL / 500 训练 / 50 独立测试

## 1. 固定协议

此文档与 `configs/sigma_qwen3_500.yaml`、`tango_iclr/controlled.py` 对应。
它替代旧 200/50/200 的下一轮受控设置，不覆盖旧结果。

| 项目 | 新设置 |
|---|---|
| 骨干 | `Qwen/Qwen3-VL-8B-Instruct`，官方这一档不是 7B |
| 训练 | 500 个 Template A prefixes |
| 验证 | 独立 50 个 Template B prefixes，仅用于选择 |
| 测试 | 独立 50 个 Template D prefixes；旧 C 保留为历史分析 |
| 干预 | decoder K，第 12–23 层，rank 8 |
| 主 teacher | local action **mean token log-probability** credit；每 block 一个 matched donor |
| donor | 从两个 reference spans 中按 prefix/block 哈希确定一个；reference 自身换另一个 |
| 种子 | 11 / 23 / 37；可以先跑 11，再跑三种子正式版本 |
| 主 controller | cross_gru，hidden 256；历史分层 token chunks 读取当前页面 chunks |
| 特征 | 第 12/18/23 层，各 8 个有序 token 均值片段；不再对所有层直接求均值 |
| 第一阶段 | 最多 20 epoch，仅训练小控制器；按验证 credit RMSE 保存 best |
| 第二阶段 | 5 epoch，主版本冻结已学 controller，只优化低秩 bank |
| 显存设置 | 默认 NF4、BF16 compute、单样本、accumulation 8；不是 16GB 保证 |
| 视觉与上下文 | 所有方法同一 max_pixels=100352、max_tokens=8192；不静默截断 |
| 主统计 | 候选准确率、答对数、paired rescue/regression；不设置 p-value 门槛 |

模型官方配置：
https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json
其中 `text_config.num_hidden_layers=36`，所以保留 12–23 而非盲目搬层号。
运行时生成 `model_lock.json` 固定实际解析出的 Hub revision。

**本轮是带截图的合成受控实验，不是原生 GUI 成功率。** 新 A/B/D 渲染器是
显式版本 v2，包含独立随机 nonce、均衡 slot 和不同布局；与旧数据不能直接作
“只换模型”比较。旧 pooled controller 会在同一份新数据、同一个 8B 骨干上重跑。
不根据 Base 的对错筛选测试样本，也不要求测试集恰好落入某个难度区间。

## 2. 针对上轮问题的代码变化

1. **保留信息的读出。** `qwen.py` 保留 probe-layer token chunks，并额外提供当前
   screenshot 的 chunks 和末尾 prompt 表示。`cross_gru` 用小型 cross-attention
   建立历史到当前观察的联系，再用 GRU 汇总历史。这里保留的是 token 顺序信息，
   不是声称完成了精确 GUI 元素检测或 2D 区域定位。
2. **第一阶段单独可查。** 输出 train/validation 的相关性、within-prefix 相关性、
   正负号准确率、balanced sign accuracy、neutral false-positive rate、argmax block
   accuracy。normalized prediction 先乘训练集 scale，再与原始 teacher credit 比较。
3. **保存第一阶段 best。** `credit_controller_best.pt` 和完整零残差 `stage1.pt`；
   policy refinement 前载入 best。验证集没有 teacher 时明确记录 train-selection，
   新 controlled runner 会生成独立 validation teacher，因此主运行使用 validation。
4. **避免第二阶段抹掉监督。** 主方法冻结 controller；`memory_joint` 保留联合训练
   作为对照。`credit_stage1_best.json` 与 `credit_stage2_epoch*.json` 可直接比较。
5. **明确正负监督。** Huber + within-prefix ranking + class-balanced signed softplus。
   新 sign term 是拟合辅助项，不是“正 gate 一定使正确概率上升”的保证。
6. **主 teacher 不偷换。** 默认仍是 local score、一个 donor。`contrastive_action`
   才计算 `s(a*) - logsumexp(s(other actions))` 的干预变化，单列消融。它也不是
   环境 expected return。两个 reference 只支持最多两个不同 donor，绝不写成三个。
7. **测试隔离。** train/smoke/ablations 不读取测试集。checkpoint 锁定后才执行 test，
   绑定 checkpoint、模型协议、测试 JSONL 和实际 PNG 哈希。修改后拒绝混用输出。
8. **控制组选择不偷用正确 credit。** shuffled / sign-flipped 使用对应变换后的
   validation credit 选其 checkpoint，同时另报其相对原始 teacher 的诊断。

主方法和 SFT/CE 使用相同的 controller/bank 架构与参数容量，但优化阶段不同：
SIGMA 有额外 teacher 成本及预训练，并在 refinement 冻结 controller；SFT 从动作监督
更新 controller/bank。请报告 teacher 成本，不要声称完全相同的监督或更新预算。

## 3. 实际运行命令

在有 CUDA 的实验机上，先安装合适的 PyTorch 和本项目依赖。默认 NF4 还需要
兼容 CUDA 的 bitsandbytes；本次没有在无 GPU 容器里假装验证它。

```bash
python -m pip install -r requirements-sigma-qwen3.txt
python -m pytest -q tests_iclr

# 数据生成：500 A + 50 B + 50 D；输出目录必须为空。
bash scripts/run_sigma_qwen3_500.sh --stage prepare --execute

# 先查看将执行的命令；未使用 --execute 不启动模型。
bash scripts/run_sigma_qwen3_500.sh --stage smoke --seeds 11

# 10 train / 8 validation 的真实 Qwen GPU 小闭环，不访问测试。
bash scripts/run_sigma_qwen3_500.sh --stage smoke --seeds 11 --execute

# 正式主对照；三种子共享同一份 teacher labels。
bash scripts/run_sigma_qwen3_500.sh --stage train --execute

# 开发集上的 17 个预定义变体。没有测试操作。
bash scripts/run_sigma_qwen3_500.sh --stage ablations --execute

# 在看测试前锁定已完成的主方法及全部消融 checkpoint。
bash scripts/run_sigma_qwen3_500.sh --stage lock --include-ablations --execute
bash scripts/run_sigma_qwen3_500.sh --stage test --execute
bash scripts/run_sigma_qwen3_500.sh --stage report --execute
```

只做主结果时，省略 ablations，lock 不传 `--include-ablations`。
要先快速验证一个种子，可以为 train/lock/test/report 全部一致地传 `--seeds 11`。
不要在同一 output 已开始后修改 seeds、像素预算、模型、损失或 Python 代码；新配置
使用新的 output。添加其余种子前最好使用独立的正式 output，防止混合测试协议。

可选更细的单 prefix GPU 合约检查（无性能含义）：

```bash
python -m tango_iclr.model_smoke \
  --config configs/sigma_qwen3_500.yaml \
  --data data/sigma_qwen3_500/train.jsonl \
  --output outputs/sigma_qwen3_contract/smoke.json
```

它检查模型类型、零残差等价、反传梯度、缓存生成和 checkpoint 写入，并记录
实测显存。它只更新一次小 bank；不是新实验成功率，也不是完整 checkpoint 重载测试。

## 4. 对比与消融

**主表**：Base、相同架构 Action SFT、candidate CE、SIGMA。

| 变体 | 唯一主要变化 | 回答的问题 |
|---|---|---|
| memory_pooled | 旧 mean-pooled GRU | 分层/片段读出是否有用 |
| memory_joint | policy 阶段继续更新 controller | credit 是否被后续优化改变 |
| memory_no_sign | sign loss=0 | 显式符号辅助是否有用 |
| memory_no_energy | energy=0 | 正则是否压低有效修正 |
| memory_ce | candidate CE 代替动作 SFT | 目标是否与候选分离更匹配 |
| memory_contrastive | 相对候选分数 teacher | teacher 是否更对齐决策 |
| memory_two_donors | 最多两个不同 reference donor | 单 donor 方差/偏差 |
| shuffle_memory | block credit 打乱 | 是否需要正确的 block 对齐 |
| flip_memory | credit 符号取反 | 是否需要正确方向 |
| unsigned | sigmoid gate | 是否需要有符号的门控 |
| uniform | 不学 credit，统一残差门控 | 是否只是额外低秩容量 |
| memory_rank4 / rank16 | rank | 容量敏感性 |
| memory_v / memory_kv | V / K+V | K 专属假设 |
| memory_early / memory_late | 0–11 / 24–35 | 层位置敏感性 |

主结果不是“只要信号有正负就算成功”。优先看：SIGMA 相对 Base 和 SFT 是否多答对，
是否 rescue 大于 regression；多种子是否同方向。50 个测试例中每一个正确数变化是
2 个百分点，应报告分子分母，不靠四舍五入制造明显提升。p 值不是实验继续门槛。

## 5. 每一阶段交付哪些结果

`outputs/sigma_qwen3_500/formal/<arm>_s<seed>/`：

- `credit_training_log.json`：第一阶段训练与选择记录（memory arms）。
- `credit_controller_best.pt`、`stage1.pt`：第一阶段 checkpoint。
- `credit_stage1_best.json`：训练/验证的真实 credit 拟合状况。
- `credit_stage2_epoch*.json`、`credit_stage2_last.json`：第二阶段变化。
- `training_log.json`：验证候选准确率及 NLL，不是环境成功率。
- `best.pt`、`last.pt`：按验证结果保存的策略 checkpoint。
- `sealed_test.json`：显式 lock/test 后才生成，逐 prefix scores 与决策。

根目录还有 `model_lock.json`、每阶段 plan、logs、run_status、selection.json、
`controlled_summary.json`。runner 是 job 级恢复；中断的 train job 会重跑该训练任务，
不是恢复 Adam 动量的训练断点。teacher 标签按原有机制逐 prefix 恢复。

## 6. 若下一轮仍不好，直接读诊断而不是追加几十个实验

- stage-1 拟合差：比较 pooled / cross_gru，检查 nonce 可读性与 teacher 方差；
  不应先增加 policy epochs。
- stage-1 好、joint 后坏：看 freeze-controller 对照，检查辅助损失权重。
- 拟合好但准确率不升：看 candidate CE / contrastive teacher / no-energy；
  “credit 识别”不等于“AB 修正方向一定正确”。
- Base 接近随机：先做开发集 current-cue oracle / OCR 可读性检查，不用最终测试调难度。
- Base 饱和：该运行只能说明保留性，不能把双方 100% 称作方法提升。

## 7. 真实 GUI 主评估没有被这套合成实验替换

本配置仅解决用户明确要求的 controlled model/data 升级。真实 AndroidWorld / MemGUI /
WorkArena 仍应使用独立的真实示范、原生动作格式与原生成功判定。不要把训练输出
`select_slot_N` 的合成模型直接当作已训练好 JSON 原生动作策略。

真实轨迹的训练沿用 `tango_iclr.real_data` / `train`，可在真实 profile 中使用相同
`cross_gru` 与冻结/联合控制策略；原生评估用 `tango_iclr.online` 或官方 MemGUI wrapper。
没有提供的数据、模拟器、服务凭据不由代码虚构。当前没有新增模型性能或 GUI 成功结果。

## 8. 本次代码验证范围

本地 CPU 回归覆盖正式扩展包和新增测试；还生成并校验了完整 500/50/50 数据与
3,600 张 PNG、6,000 个互不重复 nonce。没有完整 transformers/Qwen3 权重的 CPU
或 CUDA 实测，不声称消除了所有模型接口/显存问题。先执行真实 GPU smoke 再开正式任务。
