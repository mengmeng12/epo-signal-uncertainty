# Steps 0–4 与 local walkthrough 对齐审阅

审阅日期：2026-10-30。数据分支 `f789590`（Yuzhen），理论分支 `667e8d6`（Yadan，已包含数据分支）。GitHub 的这些成果位于 feature branches；本次从理论分支接出 Isabella 的 `feature/reliability-and-oos`，没有修改同学的分支或合并 main。

对照材料为本地 `EPO_Local_Walkthrough/minimal_validation.ipynb` 及其三张结果表。下面区分需要团队决定的口径、报告单位差异，以及已经核验一致的实现。审阅范围是已完成的 Steps 0–4 与 Step 5 的接口，不代表后续实证结果已经验证。

## 1. 缺失值处理：需要先统一的实质差异

原始数据清洗是一致的：French 缺失代码保留为 NaN。差异位于 [`replication.candidate_panel`](../src/epo_project/replication.py)：对建仓后下一月缺失的 **excess return** 执行 `fillna(0.0)`。现有数据只有一例：1943-06 建仓的 Rubbr，1943-07 回报缺失。填零 excess return 隐含该行业当月赚取 RF，不能描述成 total return 为零。

本地处理保留当时持仓，将这一期组合回报记为不可评价。随后严格要求选参窗口包含连续完整的 60 个月。1943-07 的缺失会使 1944–1948 年的年度选参窗口不可用，因此影响超过一个月。三种策略统一在 OOS EPO 可评价的月份比较。

用同一份仓库数据、同一组持仓，只改变这一规则，得到：

| 1942–2018 共同样本口径 | 月数 | Raw MVO Sharpe | MVO with Q Sharpe | OOS EPO Sharpe |
|---|---:|---:|---:|---:|
| 仓库：下一期缺失 excess return 填 0 | 924 | 0.303297 | 0.572432 | 0.750897 |
| 本地：该期不可评价，缺失窗口不选参 | 863 | 0.322111 | 0.591457 | 0.770264 |

第二行与本地三项 benchmark 结果一致。这里的 Sharpe 上升包含共同评价样本变化，不能解释成策略改进。建议正式研究保留缺失并明确报告可评价月份，另将填零作为敏感性口径；团队若决定保留填零，应记录其经济假设。不要根据 t+1 是否有回报，反过来删掉 t 时已经持有的行业，这会引入未来信息。

本次保留同学 Step 3 的代码和结果，未擅自覆盖。最终定稿时，PC diagnostics 也应同步采用约定的缺失处理；它目前使用相同的填零回报。

## 2. 行业进入 forecast universe 的时间：Step 7 前需要决定

共享 workflow 的 Step 5 要求 120 个已完成的 monthly pairs。本地 quick experiment 另加了一条：行业必须连续 60 个月满足 regression-ready 条件才进入投资池。这是本地较保守的 admission rule，不能与“选 w 需要 60 个月 candidate returns”混为一谈。

本次 Step 5 默认按共享 workflow 实现：满足 120 个连续有效 pairs、当期 signal 和 60 个月 risk history 即可进入；`minimum_ready_months=60` 可复现本地额外限制。在对应 1970–2018 实证的建仓日期中，两种规则会影响 131 个日期、354 个行业月，涉及 FabPr、Gold、Guns、Hlth、Soda、Softw。对应 2019–2025 的建仓日期没有此类差异。

团队应在 Step 7 前选定规则。改变投资池会影响 h 的 winsorization 和均值归一化，所以应从所选行业的 `u/sigma2` 重新计算 h；仅过滤已存 h 不足以保持既定定义。z 继续使用 Step 2 benchmark universe 的定义，避免同时改变信号。

## 3. Simulation 表：先统一单位和随机输入

[`simulation._mean_row`](../src/epo_project/simulation.py) 的 loss 列采用 **monthly utility bp**（月度 utility × 10,000），本地表为 **annualized utility bp**（月度 utility × 12 × 10,000）。仓库图的标题已标明月度。以 `subset=all` 为例：

| Target CV | 仓库 actual CV | 仓库 loss，bp/月 | 换算 bp/年 | 本地 actual CV | 本地 loss，bp/年 |
|---|---:|---:|---:|---:|---:|
| 0.0 | 0.000000 | 0.000 | 0.000 | 0.000000 | 0.000 |
| 0.5 | 0.539386 | 109.051 | 1308.614 | 0.459076 | 1057.773 |
| 1.0 | 1.195475 | 317.877 | 3814.529 | 0.797880 | 3101.539 |

单位统一后仍有差异：仓库使用 seed 20261023，本地使用 20261029，两者抽到的固定 h profile 不同，有限个行业的 actual CV 不等于 target CV。不要要求两张表逐项数值相等。若需要精确对照，需同时固定 h profile、均值与噪声抽样，再比较实现。

两边的定性结果相同：均匀 h 时 compression loss 约为零，所展示的异质性情景存在正损失。仓库额外给出 exposure-alignment 分组结果，本地最小表没有这一分组。每档 CV 只使用一个预先抽定的 h profile；现有 Monte Carlo standard error 以该 profile 为条件，不涵盖重新抽取 profile 的不确定性。正式报告应同时列 target CV 与 actual CV，并把 loss 单位写进列名或 notes。

仓库已做 coarse grid 加局部优化，并纳入 w=0/1 边界。本次对每档 CV 的前 30 个 draws 将 grid 从 2,001 加密到 4,001，最大 utility 变化低于 7e−17。这个检查支持所检查样本的数值稳定性；它不替代所有 draws 的完整收敛审计。

## 4. 数据 vintage：标签不同，当前历史数据一致

仓库 raw data 标注 CRSP 202608，本地为 CRSP 202609。使用相同 parser 比较截至 2025-12 的 1,194 个月后，行业缺失掩码一致，行业收益和 RF 没有超过 1e−12 的差异。行业收益最大绝对误差约 1.1e−16，属于读取后的浮点精度。因而本次 benchmark 数值差异可以由缺失规则解释，不能归因于 vintage。

正式项目继续沿用同学已提交的 raw data 和 source manifest 即可，无需为了标签一致换数据。

## 5. 已一致的关键处理

- Momentum 使用 trailing 12-month **compounded total returns**；risk model 使用过去 60 个月 excess returns 的 sample covariance，ddof=1；Q 的 5% preliminary shrinkage 保持对角线不变。
- 信号以当期 benchmark-eligible industries 为横截面，正负两边归一化为 +1/−1。建仓信息截止 t，持仓赚取 t+1 回报。
- Benchmark 采用同样的 10% predicted volatility targeting、年度选参、过去 60 个月 Sharpe 和较高 w 的 tie-break。它与后续 fixed-gamma empirical comparison 的约束不同，Step 7 不应直接复用 benchmark 的独立波动率缩放。
- 本次将本地 OLS/bootstrap 核心接入共享代码：在 1969-12、1999-12、2018-12、2025-11，按相同 inputs 和本地 admission rule 比较，s 和 u 完全一致；h 最大绝对差异低于 1.5e−13。

## 6. 后续实证还需承接的两项约定

本地已接受 **bankrupt candidate 永久退出**。本地最小实验中，w=0 与 w=0.25 在 warmup 阶段分别于 1960-06、1961-12 破产；其后 candidate returns 保持缺失，永久不再参与选参。若实际选中的 scalar 或 heterogeneous 路径破产，程序报错停止。实际路径从 1970 年的现金持仓启动，连续运行到 2025 年，2019 年不重启。共享 Step 7 尚未实现这些规则，不能只把它们看作数据缺失。本次仅交付 Step 5，后续应在 backtest 的 wealth/accounting 和 candidate selection 中显式处理。

本地 quick experiment 已经展示过 2019–2025 的实证表现。`design_decisions.md` 中原有 holdout 描述只反映 Yuzhen 当时的工作；本次已追加团队层面的更新。最终报告宜使用 subsequent evaluation，并如实说明初步结果已被查看。

## 可复查材料

- [`step5_alignment_audit.json`](step5_alignment_audit.json)：缺失月份、两种 benchmark 口径、vintage 与 oracle grid 检查。
- [`step5_verification.json`](step5_verification.json)：四个日期的 local estimator 对照和完整输出复跑检查。
- [`tests/test_reliability.py`](../tests/test_reliability.py)：独立 least-squares reference、共享日期抽样、严格连续窗口、未来扰动/截断、行业顺序与退化输入测试。

复跑审阅（从仓库根目录；local walkthrough 路径可按实际位置修改）：

```bash
PYTHONPATH=src python scripts/review_alignment.py --local-raw ../EPO_Local_Walkthrough/data/raw
```

不传 `--local-raw` 时跳过跨 vintage 比较。脚本在临时目录重建 risk inputs，不覆盖同学的 benchmark 或 simulation 结果。Step 6 的接口见 [`doc/step5_handoff.md`](../doc/step5_handoff.md)。
