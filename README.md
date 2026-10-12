# Decision Benchmark

决策模型（System One / Decision Model）四方真实 API 对比评测工程。

> **口径声明**：本仓库包含两次评测。**PubMedQA 大样本运行（500 题 × 3 轮 = 6000 次请求）
> 的 4/6 组配对区间不含零，结论可用于选型**；此前的 12 题演示集运行区间全部含零，仅用于
> 验证流程。两次的延迟均为端到端真实服务体验（含网关与网络），**不代表纯模型推理速度**，
> 且百炼走国内直连、其余三家经 OpenRouter，网络路径不对等。详见
> [选型报告](reports/pubmedqa_fourway_500x3_2026-10-12.md)的「不能外推」章节。

## 评测对象

| 服务 | 模型 / 入口 | 协议 |
|---|---|---|
| 阿里云百炼 | `decision-model-preview`（国内直连 systemone） | decision |
| 微软 Decision-1 | `microsoft/microsoft-decision-1`（OpenRouter decisions） | decision |
| Qwen3.5-9B | `qwen/qwen3.5-9b`（OpenRouter chat/completions） | chat 对照基线 |
| TypeSafe Jev 1.13 | `typesafe/jev-1.13`（OpenRouter decisions） | decision |

Qwen3.5-9B 作为**公开基础模型对照**纳入：微软公开说明 Decision-1 由 Qwen3.5-9B
后训练而来。它以关闭思考的严格 JSON Chat 方式只返回类别键，**不自报概率、不伪造
one-hot 分布**，因此其 Brier / ECE 在报告中显示为「不可用」。

## 主结果：PubMedQA 大样本（500 题 × 3 轮 = 6000 次请求）

| 指标 | 百炼 | 微软 Decision-1 | Qwen3.5-9B | Jev 1.13 |
|---|---|---|---|---|
| 首轮准确率 | 64.40% | 75.20% | 67.40% | **78.00%** |
| 95% Wilson 区间 | [60.11, 68.47] | [71.23, 78.78] | [63.17, 71.36] | [74.16, 81.41] |
| Macro-F1 | 0.4841 | 0.6042 | 0.5716 | **0.6430** |
| Brier（越低越好） | 0.5139 | 0.3553 | 不可用 | **0.3366** |
| P50 延迟 | **121.1 ms** | 525.9 ms | 1012.7 ms | 497.9 ms |
| 失败请求 | 0 | 0 | 14（上游 429） | 0 |

**4/6 组配对达到统计显著**：Jev 与微软均显著优于百炼（+13.6pp / +10.8pp）和 Qwen
（+10.6pp / +7.8pp）；**Jev 与微软之间差异不显著**（2.8pp，区间 [−6.20, +0.60]），二者统计等效。

### 选型建议

| 场景 | 推荐 | 依据 |
|---|---|---|
| 准确率优先 | Jev 或微软 | 78.0% / 75.2%，显著优于另两家；二者间无显著差异 |
| 追求最少人工 | **Jev** | 阈值 0.90 下自动处理 70.2%，人工量仅 29.8% |
| 自动错误代价高 | **微软** | 自动子集准确率 90.7%，自动集错误数最低（26 个） |
| 毫秒级实时链路 | **百炼** | P50 121ms、重复一致率 100%；须接受准确率落后约 11–14pp |
| 不建议做生产决策 | Qwen3.5-9B | 延迟最高、无原生概率，且遇上游共享池 429 限流 |

> **共性短板**：四家在 `maybe`（证据不足）类上 F1 **全部低于 0.24**。凡涉及「证据是否充分」
> 的判断都不应交给模型自动裁决，必须人工复核或规则兜底。另：百炼 `no` 类召回仅 33.7%，
> 在需识别否定结论的场景风险显著。

阈值 **0.90 在独立 dev 集选定后再到 test 验证**（非用 test 成绩反推），迁移偏差仅 0.2–4.6pp，
可直接用于生产配置。完整报告：
[Markdown](reports/pubmedqa_fourway_500x3_2026-10-12.md) ·
[14 页 PDF](reports/pubmedqa_fourway_500x3_2026-10-12.pdf)

## 参考：12 题演示集运行（仅验证流程）

| 指标 | 百炼 | 微软 Decision-1 | Qwen3.5-9B | Jev 1.13 |
|---|---|---|---|---|
| 首轮准确率 | 91.67% (11/12) | **100%** (12/12) | 91.67% (11/12) | **100%** (12/12) |
| Macro-F1 | 0.9153 | 1.0000 | 0.9153 | 1.0000 |
| P50 延迟 | **93.0 ms** | 1328.5 ms | 1975.4 ms | 839.9 ms |
| P95 延迟 | **110.1 ms** | 1643.0 ms | 4553.5 ms | 1421.1 ms |
| 正式请求成功 | 36/36 | 36/36 | 36/36 | 36/36 |
| 重复一致率 | 100% | 100% | 100% | 100% |

144 次正式请求零失败。**延迟差含网络路径差异**：百炼走国内直连，其余三家经
OpenRouter 跨境路由，不可直接比较模型速度。

## 快速开始

```powershell
cd decision-benchmark
python -m pip install -r requirements.txt

# 1) 离线演示：不需要密钥、不调用网络
python benchmark.py run --mock

# 2) 配置真实密钥
copy .env.example .env   # 填入 DASHSCOPE_API_KEY / BAILIAN_WORKSPACE_ID / OPENROUTER_API_KEY

# 3) 连接检查（每家一次真实请求）
python benchmark.py --env-override check --providers bailian,microsoft,qwen,jev

# 4) 四方正式运行（演示集，12 题，约 2 分钟）
python benchmark.py --env-override run --providers bailian,microsoft,qwen,jev \
  --rounds 3 --warmup 2 --concurrency 1

# 5) 复现主结果：PubMedQA 大样本（6000 次请求，约 75 分钟）
python benchmark.py prepare-pubmedqa          # 下载 PQA-L 并生成 dev/test 划分
python benchmark.py --env-override run --dataset data\pubmedqa\test.jsonl \
  --limit 500 --rounds 3 --warmup 10 --concurrency 1 --seed 42 \
  --providers bailian,microsoft,qwen,jev
```

> 大样本运行实测约 75 分钟、OpenRouter 侧成本约 $0.07（决策模型输出免费）。
> `--limit` 按固定种子随机抽样，不是截取前 N 条。调参只用 dev 集，
> 固定规则后不要根据 test 成绩改提示词或阈值。

微软 / Qwen / Jev **共用同一把 `OPENROUTER_API_KEY`**；百炼单独用
`DASHSCOPE_API_KEY`。详细方法学、公平性约束与指标口径见
[`decision-benchmark/README.md`](decision-benchmark/README.md)。

> **原始运行产物不入库**：`decision-benchmark/results/` 体积较大且可完整复现，
> 已在 `.gitignore` 中排除。仓库内 `decision-benchmark/example-run/` 提供了
> 一次运行的完整产物示例（report.html / summary.json / requests.csv 等）。

## 测试

```powershell
cd decision-benchmark
python -m unittest discover -s tests -v   # 34 项离线测试，不消耗 API 额度
```

测试覆盖真实 HTTP 传输、标签不泄漏、错误处理、概率校验、配对抽样与四方配对统计。

## 目录结构

| 路径 | 说明 |
|---|---|
| `decision-benchmark/` | 评测框架（provider 适配、指标、报告生成、测试） |
| `decision-benchmark/example-run/` | 一次运行的完整产物示例 |
| `reports/` | 评测报告（Markdown / PDF / DOCX） |
| `video/` | 30 秒项目介绍视频工程（HyperFrames） |
| `work/pdf-report/` | PDF 报告构建脚本 |

## 安全

- `.env` 已在 `.gitignore` 中排除，**密钥不入库**；报告与结果文件不写入认证头，
  百炼 workspace 子域在产物中脱敏为 `{workspace}`。
- 视频工程使用的微软雅黑属 Windows 版权字体，**不随仓库分发**，
  复现步骤见 [`video/README.md`](video/README.md)。

## 数据来源

公开题目可能已进入模型训练数据；科研结论应另用独立收集并标注的样本复核。
各厂商公布的性能数字均为自测，截至 2026-10 缺少独立第三方验证。
