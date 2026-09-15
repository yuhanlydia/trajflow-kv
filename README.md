# SIGMA

**SIGMA: Memory Adaptation with Signed Causal Credit for Vision-Language GUI Agents**

This is the user-designated destination for the SIGMA experiment plans: **`yuhanlydia/trajflow-kv`**, branch **`main`**.

## Start here / 执行入口

- [完整未完成实验计划：S00 + E01–E17](docs/SIGMA_ICLR2027_PENDING_EXPERIMENTS.md)
- [机器可读实验依赖、状态与产物清单](experiments/sigma_iclr2027_pending.json)

计划覆盖真实 GUI 主评估、同容量 baseline、controller/credit/KV/layer/rank/donor/energy 消融、OOD、局部 credit 到真实 rollout return 的验证、7B 复制，以及有条件的 no-hook LoRA policy distillation。每项写明设置、依赖、输出路径与对应论文图表。

## Current repository state / 当前仓库状态

The repository was empty before this plan upload. This delivery adds the experiment plan, machine-readable manifest, and this README. **It does not yet contain the original training code or the previously delivered experimental-code overlay.** No new GPU or GUI experiment results were produced in this upload.

原始代码来源是 [Yunbo-max/trajflow-kv](https://github.com/Yunbo-max/trajflow-kv)，此前交付的扩展包为 `TANGO_ICLR2027_code_overlay.zip` / `TANGO_ICLR2027.patch`。旧 TANGO 文件路径为兼容路径，论文名统一使用 SIGMA。

执行 agent 必须先完成计划中的 **S00**：迁入原始代码、应用此前代码包、核对新稿配置与实际训练/部署接口。不要把“计划已提交”解释成“完整代码已迁入”或“全部实验已可直接运行”。不要强推覆盖现有提交，也不要把 API keys、模型权重或含私密信息的截图提交到仓库。

## Execution contract / 执行约定

1. Read S00 and reconcile implementation/configuration before launching GPU jobs.
2. Freeze grouped train/validation/test manifests; never choose test tasks or donors according to model outcomes.
3. Use native real-GUI task success as the primary endpoint. Interference Chain remains an internal causal diagnostic.
4. Keep `local_score`, `candidate_value_proxy`, and `native_rollout_return` separate.
5. Record `NOT_RUN`, `BLOCKED_CODE`, `BLOCKED_DATA`, `BLOCKED_ENV`, `RUNNING`, `COMPLETE`, `FAILED`, or `DEFERRED_TEACHER_GATE` truthfully. Completion does not mean SIGMA won.
6. Preserve positive, negative, and tied outcomes. Update compact result records and provenance after each completed stage.

All listed experiments start as `NOT_RUN`; their implementation status is `AUDIT_REQUIRED`. Resolve missing dependencies in S00 and record the actual runnable entrypoint for each experiment rather than treating the plan as executable code.
