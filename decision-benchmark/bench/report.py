import csv
import html
import json
from pathlib import Path


DISPLAY_NAMES = {"bailian": "阿里云百炼", "microsoft": "微软 Decision-1 / OpenRouter",
                 "qwen": "Qwen3.5-9B / OpenRouter", "jev": "TypeSafe Jev"}

def display(name):
    return html.escape(DISPLAY_NAMES.get(name, name))


def fmt(value, percent=False):
    if value is None:
        return "—"
    return f"{value * 100:.2f}%" if percent else f"{value:.2f}"


def generate_report(directory, meta, samples, rows, summary):
    directory = Path(directory)
    (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["provider", "phase", "round", "sample_id", "task", "ok", "http_status", "latency_ms",
              "server_latency_ms", "served_model", "upstream_provider", "request_id", "error", "input_chars", "input_chars_bucket"]
    with (directory / "requests.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            safe = {k: ("'" + v if isinstance(v, str) and v[:1] in "=+-@" else v) for k, v in row.items() if k in fields}
            writer.writerow(safe)
    mock = meta["mock"]
    title = "模拟演示 · 不能用于模型排名" if mock else "Decision 模型对比报告"
    parts = [f"<h1>{title}</h1>", f"<p>{html.escape(meta['created_at'])} · {len(samples)} 个独立输入 · {meta['rounds']} 轮 · 并发 {meta['concurrency']}</p>",
             '<div class="notice">准确率仅使用第 1 轮；失败计为错误。延迟使用全部正式轮次，排除暖机；P50/P95 主指标仅含成功请求，失败率同时展示。重复请求可能受服务缓存影响。本报告比较所选服务的端到端实际体验，不代表纯模型计算速度。Qwen3.5-9B 使用关闭思考的 Chat JSON 分类；没有原生类别概率，Brier、ECE 和概率覆盖率不可用。三方的两两区间为探索性分析，未做多重比较校正。</div>']
    if mock:
        parts.append('<div class="warning">本报告由本地模拟响应生成，没有调用任何真实模型。所有数字仅用于检验报告功能。</div>')
    if meta["status"] != "complete":
        parts.append('<div class="warning">运行未完成（' + html.escape(meta["status"]) + '）。以下仅供排查：未收到结果的预定样本暂按错误计入，不能据此比较模型能力；不显示配对差异区间。请修复问题后新建完整运行。</div>')
    parts.append('<table><tr><th>服务</th><th>分类准确率</th><th>有效/总请求</th><th>失败率</th><th>P50 ms</th><th>P95 ms</th></tr>')
    for name, p in summary["providers"].items():
        parts.append(f"<tr><td>{display(name)}</td><td>{fmt(p['classification_accuracy'], True)}</td><td>{p['successful_requests']}/{p['requests']}</td><td>{fmt(p['failure_rate'], True)}</td><td>{fmt(p['p50_ms'])}</td><td>{fmt(p['p95_ms'])}</td></tr>")
    parts.append('</table>')
    for pair in summary["paired"]:
        ci = pair["ci95"]
        parts.append(f"<p>配对准确率差（{display(pair['left'])} − {display(pair['right'])}）：<b>{pair['difference'] * 100:.2f} 个百分点</b>；95% bootstrap 区间 [{ci[0] * 100:.2f}, {ci[1] * 100:.2f}]。按独立输入重采样，不将重复轮次当新样本。差异区间包含零时，不能确认谁更准确。</p>")
    for name, p in summary["providers"].items():
        parts.append(f"<h2>{display(name)} · 分任务结果</h2>")
        for task, m in p["tasks"].items():
            parts.append(f"<h3>{html.escape(task)}</h3>")
            if m["type"] == "score":
                parts.append(f"<p>有效 {m['valid']}/{m['n']}；MAE {fmt(m['mae'])}；RMSE {fmt(m['rmse'])}。仅在成功响应上计算，单位为量表等级。</p>")
                continue
            ci = m["accuracy_ci95"]
            parts.append(f"<p>准确率 {fmt(m['accuracy'], True)}，95% Wilson 区间 [{fmt(ci[0], True)}, {fmt(ci[1], True)}]；Macro-F1 {fmt(m['macro_f1'])}；Brier（各类别平方误差之和）{fmt(m['brier_sum'])}；ECE {fmt(m['ece_10_bins'])}。概率指标有效样本 {m['calibration_n']}。</p>")
            labels = list(m["confusion"])
            parts.append('<table><tr><th>真实 ↓ / 预测 →</th>' + ''.join(f'<th>{html.escape(k)}</th>' for k in labels + ['<ERROR>']) + '</tr>')
            for k, counts in m["confusion"].items():
                parts.append(f'<tr><th>{html.escape(k)}</th>' + ''.join(f'<td>{counts[c]}</td>' for c in labels + ['<ERROR>']) + '</tr>')
            parts.append('</table><table><tr><th>类别</th><th>样本数</th><th>Precision</th><th>Recall</th><th>F1</th></tr>')
            for k, c in m["per_class"].items():
                parts.append(f'<tr><td>{html.escape(k)}</td><td>{c["support"]}</td><td>{fmt(c["precision"], True)}</td><td>{fmt(c["recall"], True)}</td><td>{fmt(c["f1"])}</td></tr>')
            parts.append('</table><h4>固定自动处理比例</h4><p>按所选类别的概率排序；相同概率按样本 ID 排序。仅为测试集描述，不是已校准的生产阈值。</p><table><tr><th>目标覆盖率</th><th>实际覆盖率</th><th>自动处理数</th><th>准确率</th></tr>')
            for c in m["coverage"]:
                parts.append(f'<tr><td>{fmt(c["target_coverage"], True)}</td><td>{fmt(c["actual_coverage"], True)}</td><td>{c["accepted"]}</td><td>{fmt(c["accuracy"], True)}</td></tr>')
            parts.append('</table><details><summary>可靠性分箱（校准）</summary><table><tr><th>概率区间</th><th>数量</th><th>平均概率</th><th>实际准确率</th></tr>')
            for b in m["reliability_bins"]:
                parts.append(f'<tr><td>{b["lower"]:.1f}–{b["upper"]:.1f}</td><td>{b["count"]}</td><td>{fmt(b["mean_probability"], True)}</td><td>{fmt(b["accuracy"], True)}</td></tr>')
            parts.append('</table></details>')
        parts.append('<h3>按输入长度分组</h3><p>长度为 state 与 questions 的 JSON 字符数，不是 Token 数。</p><table><tr><th>字符数</th><th>请求数</th><th>有效数</th><th>P50 ms</th><th>P95 ms</th></tr>')
        for bucket, b in p["input_length_buckets"].items():
            parts.append(f'<tr><td>{bucket}</td><td>{b["requests"]}</td><td>{b["valid"]}</td><td>{fmt(b["p50_ms"])}</td><td>{fmt(b["p95_ms"])}</td></tr>')
        parts.append(f'</table><p>重复调用分类一致率：{fmt(p["repeat_agreement"], True)}；有效比较 {p["repeat_comparisons"]} 次。</p>')
        if p["errors"]:
            parts.append('<details open><summary>失败原因</summary><pre>' + html.escape(json.dumps(p["errors"], ensure_ascii=False, indent=2)) + '</pre></details>')
    parts.append('<h2>运行配置与复现信息</h2><pre>' + html.escape(json.dumps(meta, ensure_ascii=False, indent=2)) + '</pre>')
    style = 'body{font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#172b41;background:#f5f7fb}h1{font-size:30px}h2{margin-top:40px;color:#174b70}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}td,th{text-align:left;padding:10px 14px;border-bottom:1px solid #e2e8f0}th{background:#eaf0f6}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f6;padding:16px;border-radius:8px}.notice,.warning{padding:16px;border-left:4px solid #22859c;background:#e6f3f5;margin:20px 0}.warning{background:#fff4d9;border-color:#c89020}summary{cursor:pointer}h4{margin-bottom:0}'
    (directory / 'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>' + title + '</title><style>' + style + '</style><body>' + ''.join(parts) + '</body></html>', encoding='utf-8')
