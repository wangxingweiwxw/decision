# Decision Benchmark：微软 vs 百炼 vs Qwen3.5-9B vs Jev

比较 **OpenRouter 上的 `microsoft/microsoft-decision-1`** 与 **百炼直连 `decision-model-preview`**，
并可加入 **公开托管的 `qwen/qwen3.5-9b` 基础模型参照** 与 **TypeSafe 的 `jev-latest` 决策模型**。
Python 3.10+，仅依赖 `requests`。Windows PowerShell、macOS 和 Linux 均可使用。

本项目比较真实服务体验，包括网关、网络和服务排队，不宣称测出了模型本身的纯推理速度。

## 1. 先运行离线演示

在项目目录打开 PowerShell：

```powershell
python -m pip install -r requirements.txt
python benchmark.py run --mock
```

打开命令末尾显示的 `results\时间戳\report.html`。
`--mock` **不需要密钥、不调用网络**，响应由本地程序生成，不读取标准答案；报告醒目标注“模拟”，不能据此判断哪个模型好。

也可以双击 `run-demo.cmd`，首次运行前先安装依赖。

## 2. 配置真实 API

复制 `.env.example` 为 `.env`，用文本编辑器填入四个值：

```dotenv
DASHSCOPE_API_KEY=你的百炼密钥
BAILIAN_WORKSPACE_ID=你的业务空间ID
BAILIAN_REGION=cn-beijing
OPENROUTER_API_KEY=你的OpenRouter密钥
```

或者在运行程序的同一个 PowerShell 窗口设置：

```powershell
$env:DASHSCOPE_API_KEY = "你的百炼密钥"
$env:BAILIAN_WORKSPACE_ID = "你的业务空间ID"
$env:OPENROUTER_API_KEY = "你的OpenRouter密钥"
```

查找顺序：当前进程环境变量 → 项目 `.env` → Windows 持久化用户/系统变量。
如果旧环境变量覆盖了刚更新的 `.env`，显式加上 `--env-override`：

```powershell
python benchmark.py --env-override check
python benchmark.py --env-override run --rounds 3
```

该选项只影响本次 Python 进程，不修改系统环境变量。
现有脚本设置的密钥可以复用。项目不会把密钥写入结果；不要把 `.env` 发给他人。

固定接口：

| 服务 | 地址 | 模型 |
|---|---|---|
| 百炼 | `https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-mode/v1/systemone` | `decision-model-preview` |
| OpenRouter 微软 | `https://openrouter.ai/api/alpha/decisions` | `microsoft/microsoft-decision-1` |

双方均以 `Authorization: Bearer ...` 认证。**OpenRouter 此处不用 `/chat/completions`。**
可用 `BAILIAN_MODEL` / `OPENROUTER_MODEL` 覆盖模型名，但报告仍将服务标识为 bailian / microsoft，请检查实际返回的模型名。
默认采用 OpenRouter 平台路由并记录响应 `provider`；需要固定上游时，将 `.env` 的 `OPENROUTER_PROVIDER` 填为平台显示的 provider slug。
项目会发送 `provider.only` 并禁止 fallback；若当前 alpha 接口不接受这些路由选项，请移除此配置并记录平台实际路由。程序不擅自回退到别的模型。

## 3. 先检查连接，再跑小样本

```powershell
# 每个服务发送 1 次真实请求，会产生相应 API 用量。
python benchmark.py check

# 12 条人工编写的科学语境示例；仅用于检查流程，不能当作科研成绩。
python benchmark.py run --rounds 3 --warmup 2 --concurrency 1

# 分别诊断单个服务
python benchmark.py check --providers bailian
python benchmark.py check --providers microsoft
```

真实运行必须双方配置完整后才开始，缺密钥时在本地报错，不把“未配置”混成模型能力低。
HTTP 401/403/404/402 会提前停止提交后续样本；已经发出的请求会完成并保存。其他失败被记录，不自动重试。

常见问题：

- 401：密钥错误、失效或与服务不匹配。
- 402：检查 OpenRouter 账户余额/额度。
- 403：检查模型权限、地域、账户和访问策略。
- 404：确认模型 ID 及接口；alpha 接口可能调整，应核对官方文档。
- 429：限流；减小 `--concurrency`，增大 `--delay`，另开一次新运行。不要删除失败行。
- ProxyError：检查终端继承的代理设置与网络。程序不会修改系统代理。
- 非 JSON/缺少 `answers`：视为失败并保存可用的响应，不把聊天输出当决策结果解析。

## 4. 准备真正的科学数据

### PubMedQA

```powershell
# 从数据集作者的 GitHub 仓库下载 1000 条专家标注数据并生成划分。
python benchmark.py prepare-pubmedqa

# 若网络无法访问 GitHub，手动下载 ori_pqal.json，再执行：
python benchmark.py prepare-pubmedqa --source C:\path\ori_pqal.json

# 调参只用 dev。正式评估使用 test，固定规则后不要根据 test 成绩改提示词。
python benchmark.py run --dataset data\pubmedqa\dev.jsonl --limit 50 --rounds 1
python benchmark.py run --dataset data\pubmedqa\test.jsonl --limit 500 --rounds 3 --warmup 10 --concurrency 1
```

导入器只将 `QUESTION` 和 `CONTEXTS` 放入输入；排除 `LONG_ANSWER`，`final_decision` 只进入本地标准答案。
生成的是按标签分层、随机种子固定的 **自定义 20% 开发 / 80% 测试划分**，不是官方榜单划分。
`provenance.json` 保存来源、文件哈希和划分信息。数据使用遵循上游条款；原始论文见下方参考资料。
公开题目可能已出现在模型训练数据中，科研结论还应使用新收集并独立标注的样本复核。

### 自定义 JSONL

一行一个 JSON 对象。例如：

```json
{"id":"sample-001","task":"evidence","state":{"claim":"A的电阻高于B","record":"A=8Ω，B=3Ω，同温度测量"},"questions":{"verdict":{"type":"choice","instructions":"只根据记录判断","criteria":{"support":"支持","refute":"反驳","insufficient":"信息不足"}}},"expected":{"verdict":"support"}}
```

- `id`：整个文件中唯一；同一文章衍生多个相关样本时，需自行按文章分组划分以防泄漏。
- `task`：同一任务应保持完全相同的 `questions`（描述、选项和问题 ID）。不同题型/量表用不同 task。
- `state`：字符串、对象或数组。
- `questions`：可以有多个问题，实际调用时原样传给双方。
- `expected`：本地标准答案，**从不发送给模型**。
- `choice` 答案：选项键字符串。
- `noul` 答案：JSON `true` / `false`；统一阈值默认 0.5，`--threshold` 可覆盖，须预先设定。
- `score` 答案：从 0 开始的量表等级/数值；项目支持 2–10 级以采用文档共同的保守范围。计算 MAE/RMSE，不把它误当分类准确率或物理量。

```powershell
python benchmark.py validate data\science_demo.jsonl
python benchmark.py run --dataset data\mixed_demo.jsonl --mock
python benchmark.py run --dataset C:\path\your_test.jsonl
```

`--limit` 按固定种子随机抽样，不只截取文件前几条。

## 5. 测试设置与公平性

```powershell
python benchmark.py run --dataset data\pubmedqa\test.jsonl --limit 500 --rounds 3 --warmup 10 --seed 42
# 分开测试并发负载；不要把不同并发条件的延迟混在一起。
python benchmark.py run --dataset data\pubmedqa\test.jsonl --limit 100 --rounds 3 --concurrency 4 --delay 0.3
```

- 每轮随机排序样本，每条样本随机决定先调用哪家；双方得到相同 state/questions。
- 每个工作线程为每个服务维护独立的长连接 Session，无客户端响应缓存。
- 暖机在每个线程首次使用每个服务时进行，单独记录并排除统计。
- 并发值是同时运行的“样本对”数量，每个服务同时在途请求数不超过此值。
- 正式请求数量 = 样本数 × 轮数 × 服务数；另有最多 `并发 × 暖机数 × 服务数` 次暖机。
- 无自动重试，客户端计时包含完整 HTTP 响应体接收，排除 JSON 解析和打分。
- `--timeout` 是 requests 的连接/读取等待超时，不是强制总时长上限。
- 重复题可能命中服务端缓存；程序不声称控制了云端缓存。第一轮与后续轮次均保存在明细中。
- 记录提供方返回的 model/provider、Token 用量、request_id 和服务端 latency；不把服务端 latency 与客户端时间相减来估计网络。
- 长度分组按输入 JSON 字符数，不同模型 Token 数不强制对齐。确认输入不超过双方支持的共同上下文长度。

## 6. 报告包含什么

每次创建新的输出目录，拒绝覆盖已有目录：

| 文件 | 内容 |
|---|---|
| `report.html` | 可离线打开的中文报告；混淆矩阵、可靠性分箱、覆盖率、长度分组 |
| `summary.json` | 可用于进一步作图的指标 |
| `requests.csv` | Excel 可读的请求级状态与延迟 |
| `requests.jsonl` | 包括暖机和失败的逐请求记录、原始返回、归一化答案 |
| `dataset.jsonl` | 本次实际使用的样本快照，含本地标准答案 |
| `run.json` | 时间、模型、种子、数据哈希、并发、环境版本等 |

准确率、Macro-F1、各类召回率仅用 **第 1 轮**；失败按错误计入分母，同时报告成功响应子集准确率。
概率指标只使用有效概率；缺概率不会假定为 0 或 1。和为 0.99 等小幅舍入误差允许归一化并留警告，偏差超过 0.05 视为无效响应。
Brier 使用多分类平方误差**求和**（二分类范围 0–2），ECE 使用 10 个等宽分箱。采用“被选中类别的概率”而非厂商 proprietary confidence 作为共同校准指标。
80% 覆盖率结果是测试集按概率排序的描述，不能直接作为上线阈值；阈值须在独立开发集选择。并列概率按样本 ID 排序。
各任务准确率区间为 Wilson 95% 区间；两家准确率差使用按样本 ID 聚类的配对 bootstrap，重复轮次与同输入下多问题不当作独立输入重采样。
主延迟 P50/P95 只统计成功正式请求；同时记录所有请求延迟分位数和失败率，避免隐藏慢失败。
计时不使用 Token/秒，因为这类模型不生成自然语言序列。并发报告是该运行条件下的延迟，不宣称测到了服务的最大吞吐能力。

```powershell
# 不重新请求 API，只重建保存的报告
python benchmark.py report results\你的运行目录
```

Ctrl+C 会保存完成的请求，并在当前请求结束后生成不完整报告。运行状态不是 complete 时不能用来宣布某模型更好。
返回码：0=请求完成且无正式请求失败；2=配置/API/验证失败；130=中断。

## 7. 自动测试

```powershell
python -m unittest discover -s tests -v
```

测试使用本地 HTTP 服务和模拟响应，验证真实 HTTP 传输、标签不泄漏、错误处理、概率处理、已知指标、配对抽样、并发报告、数据导入。不访问真实模型或消耗 API 额度。

## 参考资料（2026-10-12 核对）

- 百炼决策接口：https://help.aliyun.com/zh/model-studio/decision-model-api
- 微软模型页：https://openrouter.ai/microsoft/microsoft-decision-1
- OpenRouter Decisions 请求/返回示例：https://openrouter.ai/blog/insights/what-is-jev/
- 微软官方介绍：https://commandline.microsoft.com/microsoft-decision-1-model-foundry/
- TypeSafe Jev 官方博客（一手）：https://typesafe.ai/blog/introducing-system-one-models-and-jev
- Jev API / 定价 / 端点（Vercel AI Gateway）：https://vercel.com/ai-gateway/models/jev
- PubMedQA 论文：https://aclanthology.org/D19-1259/
- PubMedQA 作者仓库：https://github.com/pubmedqa/pubmedqa
- 数据源：https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json

模型别名和 alpha 接口可能发生变化，请保留每次结果中的日期、served_model 和原始响应。本项目交付前只做了离线/本地验证，没有以真实密钥跑出任何一家模型的排名；四家（百炼 / 微软 / Qwen3.5-9B / Jev）的厂商公布指标均为各自自测，缺独立第三方验证，不能据此直接宣布谁强谁弱。Jev 当前还需 waitlist 批准且仅美国西海岸可用。


## 10. 三方对比：加入微软的后训练基础模型 Qwen3.5-9B

微软官方发布说明其 Decision-1 基于 Qwen3.5-9B 后训练：
https://commandline.microsoft.com/microsoft-decision-1-model-foundry/

通过同一 OPENROUTER_API_KEY 调用 qwen/qwen3.5-9b，无需新密钥。
默认固定 DeepInfra、BF16、禁止上游回退，require_parameters=true；可用 QWEN_PROVIDER 指定其他支持 BF16 的上游。
固定 temperature=0、max_tokens=256、reasoning.enabled=false，strict JSON schema 只返回类别键。
当前 Qwen 适配器只支持 choice 数据集；混合题型在发出请求前拒绝。
不要求模型自报概率，不伪造 one-hot 分布，因此 Qwen 的 Brier/ECE/概率覆盖率显示不可用。
state/questions 与两家决策接口相同，本地 expected 不进入请求；Chat 另有固定格式提示与 JSON schema。
每条结果保存 request_payload 供审计（不含认证头），原始响应记录真实路由和 Token。

```powershell
python benchmark.py --env-override check --providers bailian,microsoft,qwen
python benchmark.py --env-override run --providers bailian,microsoft,qwen --rounds 3 --warmup 2 --concurrency 1
```

12 条输入共 108 次正式请求 + 6 次暖机。每轮随机服务顺序，三家同次测试，准确率仅取首轮。
报告包含三组两两配对 bootstrap 区间；未做多重比较校正，作为探索性分析。
Qwen 是公开托管的 Qwen3.5-9B 基础模型参照，并非微软未公开训练前检查点的精确复现。
托管硬件、部署、精度、Chat 自回归输出与决策单次评分存在差异，不能将全部性能差归因于后训练。
三方结果必须新建运行，不能将第三家本次延迟与两家历史延迟混成同次结果。


## 11. 四方对比：加入 TypeSafe Jev（经 OpenRouter）

TypeSafe 于 2026-09-15 发布 Jev，自称首个公开的「System One」决策模型：不生成文本，
给定 state 与类型化 questions，返回每个选项的校准概率。**Jev 通过 OpenRouter 的 Decisions API 调用，
端点与微软完全一致**（`POST https://openrouter.ai/api/alpha/decisions`），走相同的 decision 协议，
因此无需 chat 适配器即可直接纳入本基准。

```powershell
# 四方连接检查（每家各发一次真实请求）
python benchmark.py --env-override check --providers bailian,microsoft,qwen,jev
# 四方正式运行
python benchmark.py --env-override run --providers bailian,microsoft,qwen,jev --rounds 3 --warmup 2 --concurrency 1
```

配置（见 `.env.example`）：

| 变量 | 说明 |
|---|---|
| `OPENROUTER_API_KEY` | **微软 / Qwen / Jev 三家共用同一把 OpenRouter 密钥**；百炼单独用 `DASHSCOPE_API_KEY`。 |
| `JEV_MODEL` | 默认 `typesafe/jev-1.13`；可用 `~typesafe/jev-latest` 跟踪最新版。 |
| `JEV_PROVIDER` | 可选，指定 Jev 的上游 provider slug 并禁用 fallback；留空则用 OpenRouter 默认路由。 |

- 无需 TypeSafe 账户或 waitlist：OpenRouter 用同一把密钥即可调用 Jev，计费归入 OpenRouter 账户。
- 四家同次测试产生 C(4,2)=**6 组两两配对 bootstrap 区间**；未做多重比较校正，作为探索性分析。
- Jev 返回原生校准概率，因此 Brier/ECE/概率覆盖率可用；Qwen Chat 基线仍不可用这些概率指标。
- 这是**真实服务体验**对比（含 OpenRouter 网关、网络、排队），不是纯模型推理速度；从中国调用 OpenRouter/百炼国际区会叠加网络延迟，延迟对比须据此解读。
- 四方结果必须新建完整运行，不能把 Jev 本次延迟与其余三家历史延迟混成同次结果。
- Jev 架构未公开、无权重、无自托管；其官方速度/成本/准确率数字均为厂商自测，缺独立第三方验证。
