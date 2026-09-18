# SIGMA ICLR 2027 — 完整单版本重写

## 打开与编译

把整个 ZIP 上传到 Overleaf，主文件选 **`main.tex`**，编译器选 **pdfLaTeX**。正文、数学、表格、两张可编辑矢量图和全部参考文献都在这个文件中。没有多个标题、多个 abstract、多个 intro，也不需要另外的 `.bib` 或 `\input` 的章节文件。

随包包含官方 `iclr2027_conference.sty`，未修改字体、页边距、行距或正文尺寸。编译两次即可解析交叉引用，不需要 BibTeX。

## 文稿结构

1. Introduction：从“最新证据仍在，但旧记忆会让决策出错”切入。
2. When Retained History Interferes：10-prefix 对照干预与 200-prefix 符号模式。
3. SIGMA：三种 utility、matched intervention、轻量历史控制器、signed residual、联合优化及部署。
4. Experiments：真实 GUI 主评估、分组切分、matched baseline、统计、消融、效率和后续 student。
5. Related Work：与 GUI memory、activation patching、ReFT 和 SPD 的准确区别。
6. Discussion and Limitations：donor 条件、冗余、测量到控制的差距及实际成本。
7. Conclusion。

附录包含完整协议、损失定义、证据边界、早期负结果及实验完成要求。

## 图表

- Figure 1：实测 10-prefix 现象图，已内嵌 TikZ。
- Figure 2：teacher measurement → credit distillation → memory adaptation，已内嵌 TikZ。
- `FIGURE_PROMPTS.md`：8 张图，每张 3 个可独立执行的构图方案，共 24 个 prompt。每张有事实锁、caption、实测/概念/缺失标签和数据需求。
- 表格已在 main.tex 中排好；未运行实验以 `NR` 标记。NR 不是 0，也不是预测值。

## 阅读版本

`main.pdf` 是完整编译稿：9 页主文，随后为声明、参考文献及附录。
单独提供的 `SIGMA_ICLR2027_main9.pdf` 是主文预览，与完整稿前九页完全相同，不是第二个写作版本。

## 有意保持的边界

这次重写没有新增 GPU 或真实 GUI 实验，也没有上传 GitHub。实测表来自已提供的结果。正式 SIGMA / recurrent controller 的真实 GUI task-success 和 matched ablation 尚待填入。论文不把“机制有证据”偷换成“已经胜过 baseline”。

已经把上版几处重要问题修正：pilot action-score 与 200-prefix candidate-value 不再混用；signed gate 不再被描述为自动保证增强/抑制；一般低秩残差不再称为正交 projector；相关论文作者与题名已核对。

详情与来源在 `EVIDENCE_AND_REVISIONS.md`。任何拟发布为匿名投稿的额外附件，先移除作者侧证据文件中的可识别仓库链接。
