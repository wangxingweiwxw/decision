# Decision Benchmark

决策模型（System One / Decision Model）四方真实 API 对比评测工程。

> **口径声明**：本仓库的评测结论来自 **12 题小样本探索性运行**，六组两两配对的
> 95% bootstrap 区间全部包含零，**统计上无法确认模型优劣**。延迟为端到端真实服务
> 体验（含网关与网络），**不代表纯模型推理速度**。请勿将本仓库数据作为选型定论或
> 模型能力排名引用。

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

## 一次真实运行的结果（n=12）

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

# 4) 四方正式运行
python benchmark.py --env-override run --providers bailian,microsoft,qwen,jev \
  --rounds 3 --warmup 2 --concurrency 1
```

微软 / Qwen / Jev **共用同一把 `OPENROUTER_API_KEY`**；百炼单独用
`DASHSCOPE_API_KEY`。详细方法学、公平性约束与指标口径见
[`decision-benchmark/README.md`](decision-benchmark/README.md)。

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
