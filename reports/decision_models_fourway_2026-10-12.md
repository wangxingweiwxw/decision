# 决策模型四方对比报告

**对比对象**：阿里云百炼 `decision-model-preview` · 微软 `Microsoft-Decision-1` · `Qwen3.5-9B`（Alibaba Cloud 开放权重基础模型） · TypeSafe `Jev`

**编制日期**：2026-10-12　|　**性质**：基于公开发布资料的能力与定位对比　|　**配套工程**：`decision-benchmark/`（已升级为四方评测）

---

## 0. 一句话结论与重要免责声明

> 决策模型（又称 "System One" 模型）是一条独立于 LLM 的新赛道：不生成文本，只对预设选项返回校准概率。四家中，**Jev 开创品类、微软 Decision-1 直接基于 Qwen3.5-9B 后训练、百炼提供直连决策接口**，而 **Qwen3.5-9B 本身是这批决策模型反复使用的开放权重底座**。

**本报告不给出"谁最强"的排名裁定。** 原因是硬性的、必须讲清楚的：

1. **所有性能数字都来自各厂商自测**（微软 36 基准、Jev 自建工作流评测等），截至 2026-10 **缺少独立第三方交叉验证**；多家媒体均强调了这一点。
2. **本项目从未用真实 API 密钥跑出四家的真实排名**。配套 `decision-benchmark` 工程仅完成离线/本地验证（mock 管线 + 单元测试）。
3. **Jev 当前为 waitlist 早期访问、仅限美国西海岸**，无 APAC/EU 区，从中国无法直接接入做真实评测。

因此下文的"对比"是**公开事实与厂商声明的结构化整理**，凡厂商自测数字均显式标注〔厂商自测〕。

---

## 1. 四方速览表

| 维度 | 阿里云百炼 decision | 微软 Microsoft-Decision-1 | Qwen3.5-9B | TypeSafe Jev |
|---|---|---|---|---|
| 厂商 | 阿里云 | 微软 | 阿里云（通义千问） | TypeSafe AI |
| 发布/可用 | 决策接口（preview） | 2026-10-09 | 开放权重基础模型 | 2026-09-15 |
| 类别 | 决策模型（System One） | 决策模型（System One） | 通用 LLM（作对照底座） | 决策模型（首个公开 System One） |
| 底座 | 未公开 | **后训练自 Qwen3.5-9B**〔微软声明〕 | 自身即底座 | 未公开（新架构，未披露）〔厂商声明〕 |
| 输出形态 | 选项校准概率 | 每选项校准概率，单次调用 | 自回归文本（本基准约束为 JSON 单标签） | choice / score / noul 三原语 + 校准概率 |
| 接口 | `/compatible-mode/v1/.../systemone` | OpenRouter `/api/alpha/decisions` | Chat Completions（本基准用严格 JSON schema） | `POST /v1/systemone` |
| 上下文 | 以接口文档为准 | 32K（文本） | 以托管方为准 | 32K |
| 定价（输入/输出） | 以百炼计费为准 | **$0.042 / 1M，输出免费**〔微软〕 | 以 OpenRouter 上游为准 | **$0.042 / 1M，输出免费**〔TypeSafe〕 |
| 开放权重/自托管 | 否 | 否（闭源托管） | **是（开放权重）** | 否（闭源托管，架构未公开） |
| 可用区域 | 中国区 / 国际区 | Microsoft Foundry（区域待列明），OpenRouter 即将 | 多家托管（如 DeepInfra） | **仅美国西海岸，waitlist** |
| 独立第三方验证 | 无 | 无〔媒体明确指出〕 | — | 仅少量上手评测，无系统性 |

> 注：百炼与 Qwen 的上下文/定价随接入形态变化，请以接入时的官方计费与接口文档为准，不在此臆造具体数值。

---

## 2. 四方逐一画像

### 2.1 阿里云百炼 `decision-model-preview`
- 通过百炼直连接口提供决策能力，端点为 `.../compatible-mode/v1/systemone`，与 Jev/微软同属"输入 state + 类型化 questions → 返回选项校准概率"的 decision 协议。
- 在本项目的**离线 mock 管线**中作为一方参与（非真实评测），真实能力需用真实密钥评测。

### 2.2 微软 `Microsoft-Decision-1`
- 2026-10-09 由微软发布，专为结构化决策任务设计；CEO Nadella 在 X 宣布。〔ref: IT之家 / 36氪 / 新浪财经〕
- **技术底座坦白说明基于 Qwen3.5-9B 后训练**，未来计划 rebase 到 MAI 与 OpenAI 自家模型。〔微软官方 Command Line 博客〕
- 自测数字〔全部为微软自测，无独立验证〕：36 基准、近 15 万题盲测取最高准确率（约 83.5%）；P50 延迟约为 GPT-6 Sol 的 1/35、比 Quyet-1.0-Large 快 4.5×；8 种扰动平均 1.3% 决策翻转（改写/乱序为 0）；安全测试 11 基准 5,250 请求。
- 内部用例：Xbox Research 标注万余条反馈，质量匹敌 GPT-6 Sol 但快 14×、便宜 200×。〔微软自述〕
- 定价 $0.042/1M 输入、输出免费；已上 Microsoft Foundry，OpenRouter 即将接入。

### 2.3 `Qwen3.5-9B`
- 阿里云通义千问的开放权重模型，**是这批决策模型反复使用的底座**：微软 Decision-1 后训练自它，AWS Strands Decider 2B 基于 Qwen3.5-2B、H2O-Lightning-4B 基于 Qwen 4B。〔ref: The Register / nowosci.ai〕
- 在本基准中**作为"决策模型 vs 通用 LLM"对照**：以关闭思考、temperature=0、严格 JSON schema 的 Chat 方式只返回类别键；**不自报概率、不伪造 one-hot 分布**，故其 Brier/ECE/概率覆盖率在报告中显示为不可用。
- 它是"公开托管基础模型参照"，**并非微软未公开的训练前检查点的精确复现**；硬件、精度、自回归输出与决策单次评分存在差异，不能把全部差距归因于后训练。

### 2.4 TypeSafe `Jev`
- 2026-09-15 发布，创始人 Diogo Almeida（前 OpenAI 研究者）；定义"System One Model"品类，不写字、只做决定。〔ref: The AI Corner / TechCrunch / Spring 官方博客〕
- 三种问题原语：**Choice**（最多 255 选项，返回概率分布+置信度）、**Score**（有序量表打分）、**Noul**（是非，返回 0–1 概率）；多问题对同一 state 并行求值。
- 训练方法 RLCD（为校准决策的强化学习），主打"置信度即产品"（高置信=高准确率）。
- 自测数字〔TypeSafe 自建工作流评测，官方自认偏乐观〕：速度最高 193.6×、成本最高 444.6×；端到端 70–500ms vs LLM 3–329s；准确率约 67–68%，与 GPT-5.6 Luna 持平、低于 Sol 约 6 点。
- **"永不幻觉"= 结构/schema 合法有数学保证，不等于内容一定正确**（官方承认可"自信地答错"）。
- 接入限制：waitlist 早期访问、**仅美国西海岸**、输入 $0.042/1M、输出免费、32K 上下文、仅文本（不支持图像）、最多 255 选项、未公开非英文准确率。

---

## 3. 两条值得注意的交叉事实

1. **$0.042/1M 输入 + 输出免费** 这一价格，微软 Decision-1 与 TypeSafe Jev **完全一致**——决策模型品类正在形成趋同定价。
2. **Qwen3.5-9B 是这波决策模型的"公共底座"**：微软 Decision-1 直接后训练自它。因此本四方对比里，Qwen3.5-9B 既是"被对比方"，也是微软一方的"前身"，解读差距时需格外注意这层血缘关系。

---

## 4. 配套工程：`decision-benchmark` 已升级为四方

本次已把评测工程从三方扩展为**四方**（改动均为最小、可回退）：

- `bench/providers.py`：新增 `jev` provider（decision 协议，`POST /v1/systemone`，独立密钥 `TYPESAFE_API_KEY`，无 OpenRouter 头）。
- `bench/report.py`：新增 Jev 显示名。
- `bench/cli.py`：`--providers` 帮助与总描述更新为四方。
- `.env.example`：新增 `TYPESAFE_API_KEY` / `JEV_MODEL` / `TYPESAFE_BASE_URL`。
- `tests/test_four_way.py`：**新增 9 项测试**（Jev 构造、缺密钥报错、base_url 校验、无 OpenRouter 头、原生概率保留、C(4,2)=6 组配对、四方 mock 全流程等）。
- `run-fourway.cmd`：四方一键脚本。
- `README.md`：新增第 11 节（Jev 接入、限制、方法学）并更新参考资料与免责声明。

**验证结果**：`python -m unittest discover -s tests -v` → **35 passed**（原 26 + 新 9），无回归；四方 mock CLI 全流程跑通并生成含 6 组配对的报告。该跑通为 **mock 模拟**，仅验证管线，不代表模型能力。

### 要跑真实四方评测，需满足：
```powershell
# .env 填入四家密钥（Jev 需 waitlist 批准）
python benchmark.py --env-override check --providers bailian,microsoft,qwen,jev
python benchmark.py --env-override run   --providers bailian,microsoft,qwen,jev --rounds 3 --warmup 2 --concurrency 1
```
在拿到真实结果前，任何"四家排名"都不成立。

---

## 5. 方法学与局限

- 准确率仅取第 1 轮，失败计为错误；延迟 P50/P95 仅统计成功正式请求，另记全部请求分位数与失败率。
- 四家两两配对使用按样本 ID 聚类的 bootstrap 区间（C(4,2)=6 组），**探索性、未做多重比较校正**；区间含零即不能判定谁更准。
- 比较的是**端到端真实服务体验**（含网关/网络/排队），非纯模型推理速度。Jev 仅美西部署，异地调用叠加跨太平洋往返，延迟解读须谨慎。
- Qwen Chat 基线不产出原生类别概率，概率类指标（Brier/ECE/覆盖率）不可用。
- 所有厂商公布指标为各自自测，缺独立第三方验证；公开题目可能已进入训练集，科研结论应另用独立标注样本复核。

---

## 6. 来源

- 微软官方：commandline.microsoft.com/microsoft-decision-1-model-foundry/
- 微软报道：IT之家、36氪、新浪财经、搜狐、星島頭條（2026-10-10）
- 微软底座为 Qwen3.5-9B / 决策模型生态：The Register、nowosci.ai、innfactory.ai（2026-10-09/10）
- TypeSafe Jev 官方：typesafe.ai/blog/introducing-system-one-models-and-jev；Spring 官方博客（2026-09-21）
- Jev 评测与限制：The AI Corner、devlery.com、modelscope.cn、Vercel AI Gateway（2026-09）
- 决策模型品类与竞品：Perplexity 汇总（Amazon/Cloudflare/OpenAI 等，2026-10-02）

> 媒体报道为辅助线索；厂商自测数字已逐项标注，不作为独立结论。
