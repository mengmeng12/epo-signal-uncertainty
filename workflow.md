# Project workflow 三人分工与交付表

研究问题：什么时候一个 scalar 能保留决策相关的不确定性信息，什么时候即使最佳 scalar 仍有损失？模拟识别机制；实证检验可实施代理，不直接识别真实 Λ。

当前状态：**初始目录已建立；所有研究任务均待三人逐步完成。** `end_to_end_pipeline.ipynb` 保持空白，`src` 和 `data` 仅有 `.gitkeep`。下表的代码入口、子目录和输出文件都是计划，并非已经存在的实现或结果。

| Step | 负责人 / 复核人 | 应做什么 | 必须生成的结果 | 完成标准 | 依赖 |
|---|---|---|---|---|---|
| S00 Shared conventions | 三人共同 / 三人共同 | 确认proposal、Python环境、共享参数、数据时点和结果命名；决定是否曾查看2019–2025。 | 计划创建：`src/epo_project/config.py`、环境依赖文件、数据字典与holdout使用记录 | 共同w、γ=3、窗口/费用/样本期与proposal一致；形成书面共识。 | 无 |
| S01 Prepare industry returns | Yuzhen Chen / Shuchen Meng | 下载49-industry月度收益和RF；清理缺失码与单位；记录vintage和hash；生成逐月可用行业。<br>入口：`data.prepare_returns` | `data/processed/returns.csv`<br>`data/processed/eligibility.csv`<br>`data/raw/source_manifest.json` | 同一日期行业唯一；收益为小数；缺失不补未来；warm-up规则可追溯。 | S00 |
| S02 Build momentum and risk inputs | Yuzhen Chen / Yadan Zhan | 实现12月momentum、60月风险、固定5%预收缩；缓存时点与行业顺序。<br>入口：`data.build_inputs` | `data/interim/momentum.csv`<br>`data/interim/risk_models.npz`<br>`data/interim/risk_index.csv` | momentum两条腿+1/-1；每个Q正定且对角不变；训练截止日期不超过t。 | S01 |
| S03 Validate one EPO benchmark | Yuzhen Chen / Shuchen Meng | 仅Equity 1；raw MVO、5% MVO、scalar EPO；年度过去60月SR选w；PC在t形成、t+1评价。<br>入口：`replication.run` | `data/results/replication_table.csv`<br>`data/results/problem_pc_diagnostics.csv`<br>`data/results/problem_pc.png` | 1942–2018在历史允许时评价；记录与论文实现差异；绝不宣称逐点复刻。 | S02 |
| S04 Identify scalar compression loss | Yadan Zhan / Shuchen Meng | 实现已知Σ/Λ的Gaussian DGP、heterogeneous与最佳scalar；分开参数选择损失和L；alignment为条件描述。<br>入口：`simulation.run` | `data/results/simulation_draws.csv`<br>`data/results/compression_loss.csv`<br>`data/results/compression_loss.png` | h=1时L≈0；oracle不读取本次μ；数值搜索含边界/零仓位极限；1000配对draws/CV。 | S00 |
| S05 Estimate forecasts and relative uncertainty | Shuchen Meng / Yuzhen Chen | 120对已完成观测OLS；共同日期12月block、200次refit；当前z下预测均值方差/资产方差，5/95截尾、floor、归一化。<br>入口：`reliability.run` | `data/interim/forecast_reliability.csv`<br>`data/results/reliability_diagnostics.csv` | h均值1且正；全零方差回退h=1；信号和h只能使用≤t信息；不加残差方差。 | S02 |
| S06 Validate the uncertainty estimator | Yadan Zhan / Shuchen Meng | Yadan定义一个固定线性DGP；复用Shuchen同一估计器，检查均值误差、bootstrap方差和覆盖率。<br>入口：`simulation.validate_estimator` | `data/results/estimator_calibration.csv`<br>`data/results/estimator_calibration.png` | 所有DGP参数及seed留档；不是人为添加multiplicative noise替代估计器；市场收益方差不等于均值误差。 | S05 |
| S07 Run the matched empirical portfolios | Shuchen Meng / Yuzhen Chen | 共同s/Q/γ=3；每年scalar过去60月净CE选共同w；heterogeneous只替换h；10bp成交+50bp年化空头费。<br>入口：`backtest.run` | `data/results/annual_selection.csv`<br>`data/results/weights.csv`<br>`data/results/matched_returns.csv` | 无rho搜索；同一w/行业/日期；主结果不独立vol target；冻结2019–2025设计；漂移调仓及资金账可复核。 | S05 |
| S08 Evaluate economic value and uncertainty | Shuchen Meng / Yadan Zhan | 净ΔCE和1000次paired block bootstrap 95%CI；汇总SR、vol、95%ES、drawdown、turnover、gross。<br>入口：`evaluation.run` | `data/results/performance_table.csv`<br>`data/results/delta_ce_bootstrap.csv`<br>`data/results/equity_curves.png` | 主block=12月，6/24敏感性；配对重采样；不显著不等价；L与实证ΔCE分开解释。 | S07 |
| S09 Check deployment sensitivity | Shuchen Meng / Yuzhen Chen | 复用主分析w；统一10%事前vol规则后gross≤3；报告cap频率；其他费用/因子/置换为可选后续。<br>入口：`evaluation.deployment_sensitivity` | `data/results/deployment_sensitivity.csv` | 不重新给heterogeneous调w；不把受约束结果套用无约束理论最优性。 | S07 |
| S10 Assemble report evidence | All three / All three | Yuzhen写数据/复现；Yadan写理论/模拟；Shuchen写实证/结论；各自准备对应presentation。<br>入口：`reporting.run` | `data/results/result_manifest.json`<br>`data/results/report_summary.md` | 结果路径、参数、vintage、seed、Git commit可追溯；三人交叉复核；不将合成数据结果写成实证。 | S03, S04, S06, S08, S09 |

## 交接顺序

1. 三人先创建并确认共享配置、环境依赖、日期/单位/列名约定和 holdout 使用记录。建议将共享配置放在未来的 `src/epo_project/config.py`，数据说明放在未来的 `data/README.md`。配置改动需在报告中说明理由。
2. Yuzhen 完成 S01–S02 后交付数据和风险接口；Yadan 可同时开展 S04，不需要等待市场数据。
3. Shuchen 完成可复用的预测/不确定性估计器；Yadan 用该模块完成 S06。S06 的历史依赖表示接口交接，不表示模拟使用市场估计的真实均值。
4. Yuzhen 完成 S03；Shuchen 完成 S07–S09。两人一起审计历史选择、漂移成交、现金融资和成本。
5. 三人整合 S10：Yuzhen 负责数据与复现，Yadan 负责理论与模拟，Shuchen 负责可实施实证与结论。三人共同检查图表和文稿。

## 分支与协作

- Yuzhen：`feature/data-and-replication`。
- Yadan：`feature/theory-and-simulation`。
- Shuchen：`feature/reliability-and-oos`。
- 小步提交并开 PR；由表中的复核人检查后合并。后续填充共享 notebook 时只放调用与叙述，主要计算写入各自模块，减少冲突。
- 改动应附测试命令、输出路径及数据版本。当前 `data` 除 `.gitkeep` 外默认不提交；三人开始开发时再明确原始数据、缓存和精选图表的版本管理规则。
- 协作者由仓库所有者添加；本次不发送邀请。不预设未经三人确认的截止日期。

## 结果解释边界

理论 L 的非负性来自给定模型和最优性；真实净 ΔCE 可能为负。共同 w 的实证比较只评估既定可实施规则，不证明战胜全部 scalar 规则。10% vol / gross 3 属于部署敏感性，主实证使用固定 γ。最终表格必须分别标注历史样本 1970–2018 和扩展样本 2019–2025；若后者参与设计，不能称 untouched holdout。

## 开始开发前共同确认的数据约定

- 日期使用ISO月末；收益率用小数表示，明确simple return、RF与excess return，缺失码保持缺失。
- formation date = t，持有期收益 = t+1；回归只使用 u+1≤t 的已完成配对。
- covariance矩阵必须附日期及行业顺序；两种策略必须共享可用行业、信号、Q、w及费用规则。
- 建议未来在 `data` 下建立 raw、interim、processed、results；目前不创建或填充这些子目录。
- 每个结果记录输入版本、代码commit、参数、seed和生成日期。共享接口确定后，再编写实际数据schema和校验。
