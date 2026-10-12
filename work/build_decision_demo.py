import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'decision-benchmark' / 'data'
root.mkdir(exist_ok=True)
questions = {"verdict": {"type": "choice", "instructions": "只根据给出的记录判断 claim。区分反驳与信息不足；不补充未提供的实验结果。", "criteria": {"support": "记录直接支持该结论", "refute": "记录直接反驳该结论", "insufficient": "记录不足以判断该结论"}}}
examples = [
    ("physics", "样品 A 的电阻大于样品 B。", "同温度下，A 电阻 8 Ω，B 电阻 3 Ω。", "support"),
    ("physics", "样品 A 的电阻大于样品 B。", "同温度下，A 电阻 2 Ω，B 电阻 7 Ω。", "refute"),
    ("physics", "样品 A 的电阻大于样品 B。", "测得样品 A 电阻 8 Ω，未测量 B。", "insufficient"),
    ("physics", "本次模拟已满足残差收敛标准。", "残差收敛标准为低于 1e-6；最终残差为 2e-8。", "support"),
    ("chemistry", "在记录的条件下，催化剂 X 的产率高于 Y。", "相同反应条件下，X 的产率 80%，Y 的产率 60%。", "support"),
    ("chemistry", "样品中检测到氯离子。", "检测报告明确记录：氯离子检测结果阴性。", "refute"),
    ("chemistry", "催化剂 X 比 Y 更耐长期使用。", "只记录了 X 与 Y 的首次反应产率，没有长期测试。", "insufficient"),
    ("chemistry", "反应在 25 摄氏度完成。", "实验日志：恒温 60 摄氏度，反应结束；未进行 25 摄氏度实验。", "refute"),
    ("biology", "处理组平均细胞数高于对照组。", "处理组平均细胞数 120，对照组平均细胞数 80。", "support"),
    ("biology", "实验使用了小鼠。", "实验记录明确说明仅使用酵母，未使用动物。", "refute"),
    ("biology", "处理可提高细胞存活率。", "记录只给出了处理组的荧光强度，没有存活率数据或对照。", "insufficient"),
    ("biology", "处理会改变基因表达量。", "本实验只测量了培养液 pH，没有测量基因表达。", "insufficient"),
]
with (root / 'science_demo.jsonl').open('w', encoding='utf-8') as f:
    for i, (field, claim, record, gold) in enumerate(examples, 1):
        item = {"id": f"demo-{i:03d}", "task": f"demo_{field}_evidence", "state": {"claim": claim, "record": record}, "questions": questions, "expected": {"verdict": gold}, "metadata": {"source": "Handwritten synthetic smoke-test, not research benchmark"}}
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
mixed = {"route": {"type": "choice", "instructions": "按明确规则判断后续操作：已收敛且检查通过可接受；明确未收敛要重算；信息不全要人工检查。", "criteria": {"accept": "接受结果", "retry": "重算", "review": "人工检查"}}, "converged": {"type": "noul", "instructions": "记录是否明确说明计算已收敛？"}, "completeness": {"type": "score", "instructions": "只按提供的记录给完整性评分。", "criteria": ["没有状态和检查信息", "只有收敛状态，缺少检查结果", "有收敛状态和检查结果"]}}
with (root / 'mixed_demo.jsonl').open('w', encoding='utf-8') as f:
    for i, (state, route, converged, score) in enumerate([("计算已收敛，所有检查通过。", "accept", True, 2), ("计算明确未收敛，检查未通过。", "retry", False, 2), ("计算已收敛，未提供检查结果。", "review", True, 1), ("任务记录为空。", "review", False, 0)]):
        f.write(json.dumps({"id": f"mixed-{i}", "task": "simulation_qc", "state": state, "questions": mixed, "expected": {"route": route, "converged": converged, "completeness": score}, "metadata": {"source": "Synthetic smoke test only"}}, ensure_ascii=False) + '\n')
