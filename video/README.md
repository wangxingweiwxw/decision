# 项目介绍视频（HyperFrames）

30 秒 1080p 项目介绍片，数据全部来自 `decision-benchmark` 的真实评测运行
（`results/20261012T101454_247943Z`）。

## 为什么仓库里没有字体和成品 MP4

- `fonts/` 下用的是 **微软雅黑（msyh.ttc / msyhbd.ttc）**，属 Windows 系统版权字体，
  **不随仓库分发**。渲染前需本机自备。
- `renders/` 下的 MP4 与 `audio/*.wav` 中间文件体积大且可完整复现，同样不入库。

## 本地复现步骤

```powershell
# 1) 准备中文字体（Windows 自带）
mkdir video\intro\fonts
copy C:\Windows\Fonts\msyh.ttc    video\intro\fonts\
copy C:\Windows\Fonts\msyhbd.ttc  video\intro\fonts\

# 2) 生成背景音乐（纯合成，无第三方素材）
#    见 audio/README.md 中的合成参数；或直接用 bgm.m4a（已入库）

# 3) 校验与渲染
node <HYPERFRAMES_RUNNER> inspect .
node <HYPERFRAMES_RUNNER> render .
```

渲染产物落在 `renders/`，1920×1080 / 30fps / 30s / H.264。

## 结构

| 路径 | 说明 |
|---|---|
| `index.html` | composition 本体（GSAP 时间轴 + 6 幕场景 + 音频轨） |
| `assets/` | 画面内引用的图片素材 |
| `audio/bgm.m4a` | 背景音乐（程序合成，Am–F–C–G 进行） |
| `vendor/gsap.min.js` | 动画库 |

## 内容口径

视频第 5 幕明确标注本次评测的局限：12 题小样本、6 组配对区间全部包含零、
统计上无法确认模型优劣；延迟页标注百炼走国内直连、其余三家经 OpenRouter，
网络路径不对等。不得将该片作为模型能力排名使用。
