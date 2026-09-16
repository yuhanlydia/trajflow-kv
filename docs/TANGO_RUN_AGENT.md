# TANGO ICLR 2027 — Agent 执行说明

## 交付状态

目标仓库：`Yunbo-max/trajflow-kv`，分支 `master`。本包基于已读取的提交
`131014ab50d6bbccff4c5ecb1332679256fa40b7`，新增独立的 `tango_iclr/`，不覆盖旧实现或旧结果。
本轮 GitHub 写入请求返回 403，因此这个包**没有推送到远端**。先应用补丁，再由具有写权限的执行 agent 提交。

本地已运行 CPU 单元测试、模拟可微模型训练和 LaTeX 编译；**没有运行真实 Qwen GPU 训练或 GUI 性能实验**。
源码中的原生模型/环境调用仍须先通过实验机 smoke。不能把 CPU 测试当作性能结果。

## 1. 应用代码

在原仓库工作区无冲突时执行；不要丢弃其他 agent 尚未提交的更改：

```bash
git status --short
git pull --ff-only
git switch -c tango-iclr2027-release
git apply --check /absolute/path/TANGO_ICLR2027.patch
git apply /absolute/path/TANGO_ICLR2027.patch
# 或将 code_overlay.zip 中的文件覆盖到仓库根目录；本补丁只有新增文件。
```

先装与 CUDA 对应的 PyTorch，再安装 `requirements-tango-iclr.txt`。Python 建议 3.11/3.12。
官方 MemGUI runtime 保持自己的 Python 3.12 / uv.lock 环境，不与训练环境强行混装。

```bash
python -m pip install -r requirements-tango-iclr.txt
python -m pytest -q tests_iclr
python -m tango_iclr.preflight --output outputs/tango_preflight.json
```

无 CUDA 会返回非零状态，这是正确的阻塞报告。AndroidWorld/MemGUI 的环境宿主机需要可用 KVM；
模型 GPU 与环境宿主机可以不同，远端 AndroidWorld 使用已有 HTTP 服务。
本包不会自动启动 privileged Docker、改账号权限、安装 CUDA 或消耗评估 API。

## 2. 数据必须先准备好

本包不是空输入即可产生论文结果的黑盒。Git 仓库没有完整截图、模型权重、受控账号或原生服务器。
执行机须提供：训练、验证、测试 JSONL及截图，和测试任务 manifest。不要用测试任务轨迹训练后再报告全套零样本成绩。
公开 benchmark trajectories 也可能属于测试任务；公开不代表可以用来训练。

转换已有逐候选 JSONL：

```bash
python -m tango_iclr.prepare \
  --train /data/locked_train.jsonl \
  --validation /data/locked_validation.jsonl \
  --test /data/locked_test.jsonl \
  --image-root /data \
  --output data/tango_release
```

`task_id` 应标识整个 task/episode 分组，不能把每个 prefix 当成独立任务。
旧格式中的 `task_family` 和 `seed` 会保守地映射为同一实例分组；跨 family 泛化请用 `--group-key family`。
Train/val/test必须有不重叠分组；不能在最终测试上调整难度、rank、层数或 gate 阈值。

规范示例（路径和数值仅展示 schema，不是可用实验数据）：

```json
{
  "prefix_id": "train_task17_step4",
  "task_id": "train_task17",
  "family": "contacts",
  "split": "train",
  "instruction": "The benchmark instruction",
  "image": "/data/task17/step4.png",
  "history_images": ["/data/task17/step2.png", "/data/task17/step3.png"],
  "history_actions": ["{\"action_type\":\"click\",\"x\":100,\"y\":200}", "{\"action_type\":\"wait\"}"],
  "candidates": ["candidate action 0", "candidate action 1"],
  "target_index": 0,
  "logged_action_index": 1,
  "trajectory_return": 0.0,
  "action_values": [0.75, 0.25],
  "critical_step": true
}
```

`action_values` 必须来自有记录的 continuation evaluator/rollout，不得给未执行的候选编造 Q。
`target_index` 是监督动作；`logged_action_index` 是行为策略真正执行的动作，两者不可混淆。
Global-return 与 successful-only CE 要求真实 logged return；缺少时会报错，不能用 max-Q 冒充轨迹结果。
没有 Q 时仍能运行 CE/局部分数 teacher，但不能声称完成 action-advantage baseline。
参考动作只用于训练/离线评分，不进入在线 policy prompt。

## 3. 实验入口

编辑 `configs/tango_iclr_release.yaml` 的模型路径、数据路径、输出目录和 native manifests。
模型 revision 必须在首次下载后固定。默认是 3B、middle K 12–23、rank8、GRU128、单GPU串行。
16GB/24GB 上先用公共的较小图像预算；OOM 时同时调整全部比较方法的预算，并记录更改。
不承诺某个分辨率或多图历史必然适合所有显卡。

```bash
# 只打印完整作业计划，不执行训练
bash scripts/run_tango_iclr.sh --profile all

# 先运行10-prefix teacher + 一轮训练/评估，检查真实模型路径
bash scripts/run_tango_iclr.sh --profile smoke --execute

# Smoke成功后完整执行，包括对照、消融、student训练及其评估
bash scripts/run_tango_iclr.sh --profile all --execute --continue-on-error
```

模式：`core`、`ablations`、`distill`、`all`。`distill` 包括其训练依赖，并非跳过 teacher。
`all` 中 CPU/GPU子进程逐个执行；已有合格 artifact 且输入/代码指纹一致才复用。
缺数据/依赖记为 BLOCKED，训练或原生运行失败记为 FAILED。查看：

- `outputs/tango_iclr2027/suite_plan.json`
- `outputs/tango_iclr2027/suite_status.json`
- `outputs/tango_iclr2027/logs/`

`--continue-on-error` 只让其他独立任务继续，**不会把失败变为成功**。

## 4. 对照与消融

主对照：Base、CE、successful-only CE、global-return、reference-weighted action advantage、
无 gate 低秩变换、unsigned gate、shuffled credit、sign-flipped credit、TANGO memory gate。
默认训练 seeds=0/1/2。TANGO 先拟合 measured credit，再以相同候选策略监督训练 gate + 低秩变换；
不是仅凭 credit regression 就假定 policy 会改善。

消融：MLP/GRU/小 Transformer、去 energy、去 credit auxiliary、rank4/8/16、
K/V/K+V、early/middle/late、单/多 donor、local-score/candidate-value teacher。
层或 K/V改变时重新算 teacher labels。固定 gradient-subspace 是明确命名的 **SPD-style GUI control**，
不是已完整复现 SPD 的 self-generation + LoRA SFT。Unsigned trust control 也不是 FocusMem 官方复现。
如需论文强 baseline 的完整官方复现，须额外接入其公开代码并报告匹配预算；本包没有冒充该结果。

Distillation：同模型 LoRA rank16，比较 KL、centered residual、hard-action CE。
每种 student 都有独立 offline/native online job，不带 teacher hook。
只在 teacher 原生成功率确有改善后解释“保留 teacher gain”；纯 merge 不算学习。

## 5. 真实 GUI 测试

### AndroidWorld

使用现有 `trajflow_kv.androidworld_http.AndroidWorldHTTPClient`，原服务管理 task initialization 与 native score。
manifest每行一个真实注册任务实例：

```json
{"case_id":"aw_task17_instance0_decode0","task_id":"aw_task17","seed":0,"decode_seed":0,"backend":"androidworld_http","base_url":"http://127.0.0.1:5000","task_type":"REGISTERED_TASK_TYPE","task_idx":0,"success_threshold":1.0}
```

把 REGISTERED_TASK_TYPE 换为服务器**实际注册**任务，不编造 task 名称。
`task_idx` 才是服务器实例编号；修改 decode_seed 不等于重新采样环境参数。
同一 manifest用在所有方法。API健康、固定正确脚本、setup/teardown与截图都正常才开始。
不得复用旧 Bluetooth/Wi-Fi hard-coded候选坐标作为主要 free-action成绩。

### BrowserGym / VisualWebArena / WorkArena

`online.py` 通过官方 BrowserGym 注册环境和 coordinate action set 运行。
需要原生站点、相应注册模块和必要账号；WorkArena 服务不是本包自带。
manifest以实际registry输出为准，包含 `backend=browsergym, registration_module, env_id, seed, task_id, case_id`。
对已安装 MiniWoB 可用 `browsergym.miniwob` 和实际注册的 `browsergym/miniwob.click-color` 做 plumbing smoke；
这不是主要 memory benchmark，也不替代真实长流程实验。

### MemGUI-Bench

MemGUI 有自己的 environment/action/evaluation protocol，**不臆造其 reset API**。
保持官方 runtime，使用本包的串行兼容模型 endpoint：

```bash
export MEMGUI_DIR=/absolute/path/MemGUI-Bench
export TANGO_MODEL=/absolute/path/Qwen2.5-VL-3B-Instruct
export TANGO_CHECKPOINT=/absolute/path/trajflow-kv/outputs/tango_iclr2027/memory_s0/best.pt
export TANGO_LOG_ROOT=/absolute/path/results/memgui_tango_s0
export TANGO_TASKS=ALL
bash scripts/run_tango_memgui.sh
```

执行前先在官方仓库配置 `.env`，启动健康 Docker/KVM backend。官方 evaluator 的模型和 key 与 policy endpoint 分离。
`qwen3vl` 是官方 wrapper 名称，不表示本包把 3B 模型冒充成 8B。
首个任务应检查请求是否真的包含**历史截图与当前截图**；wrapper只发当前图时本方法没有历史可控，不能继续报告TANGO memory结果。
native MemGUI wrapper、图像格式、上下文长度兼容性须在实验机验证；本地只验证了 endpoint代码。
MemGUI官方 pass@k遵循官方attempt协议，不能把3个随机seed的均值叫pass@3。
当前总runner原生调度AndroidWorld/BrowserGym；MemGUI完整矩阵由独立官方wrapper脚本逐checkpoint运行。

## 6. 真正的 trajectory-return teacher

`python -m tango_iclr.rollout_credit --help` 提供 reset/replay + one-decision KV patch 的配对采样。
输入要求 `environment, replay_actions, decision_state_hash, matched_donors`。
相同prefix恢复失败则BLOCKED；浏览器时间戳/动画造成截图不一致时，应先增加原生状态验证，不可放宽成任意不同状态。
该操作只改变下一次决策，然后按base policy继续；不是永久篡改后续所有cache。
默认teacher的 candidate-value proxy并不等于这种 native-return teacher，不能互换命名。
已有 paired-return标签可直接作为训练输入；总runner默认自动生成的是候选proxy，不自动重新初始化所有远端分支。

## 7. 统计与结论

```bash
python -m tango_iclr.report \
 --method outputs/tango_iclr2027/memory_s0/online_0/episodes.jsonl \
 --baseline outputs/tango_iclr2027/ce_s0/online_0/episodes.jsonl \
 --output outputs/tango_iclr2027/paired_tango_vs_ce.json
```

主指标 native task success；离线 candidate Top-1、label margin、memory sign不替代成功率。
按task聚类，不把同prefix的多个block当独立样本。报告rescues/regressions、cluster bootstrap CI、覆盖率和成本。
默认report拒绝缺配对或混入infra错误；必须先修复/重跑，或预注册单独coverage分析。
不隐藏negative结果，也不要求先验指定的方法排序一定成立。

## 8. 提交与交接

完整执行日志、environment/model revisions、split_manifest、suite_status、失败原因及真实原生结果一起提交。

```bash
git add tango_iclr tests_iclr configs/tango_iclr_release.yaml scripts/run_tango_iclr.sh scripts/run_tango_memgui.sh requirements-tango-iclr.txt docs/TANGO_*.md paper_iclr2027_tango
git commit -m "Add TANGO ICLR2027 evaluation, ablations and manuscript package"
git push -u origin tango-iclr2027-release
```

论文写作包中的现有历史数值来自用户提供的项目结果和仓库记录，未在本轮复算。
真实 TANGO 相比 CE 的显著提升、跨模型收益和 no-hook 保留率仍必须等新实验。
