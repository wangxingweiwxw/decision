from itertools import combinations
import math
import random
from collections import Counter, defaultdict
from statistics import mean


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def wilson(correct, total):
    if not total:
        return None
    z, p = 1.95996398454, correct / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [center - width, center + width]


def label(value):
    return str(value).lower() if isinstance(value, bool) else str(value)


def classification_metrics(records, labels):
    """Missing/invalid responses count as incorrect; no fabricated probability."""
    n = len(records)
    successful = [r for r in records if r["pred"] is not None]
    correct = sum(r["pred"] == r["gold"] for r in successful)
    matrix = {gold: {pred: 0 for pred in labels + ["<ERROR>"]} for gold in labels}
    for r in records:
        matrix[r["gold"]][r["pred"] if r["pred"] is not None else "<ERROR>"] += 1
    per_class, f1s = {}, []
    for k in labels:
        tp = matrix[k][k]
        fn = sum(matrix[k].values()) - tp
        fp = sum(matrix[other][k] for other in labels if other != k)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        f1s.append(f1)
        per_class[k] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    calibrated = [r for r in successful if r["probabilities"] is not None]
    brier = mean(sum((r["probabilities"][k] - float(r["gold"] == k)) ** 2 for k in labels)
                 for r in calibrated) if calibrated else None
    bins = []
    for i in range(10):
        subset = [r for r in calibrated if min(9, int(r["pred_probability"] * 10)) == i]
        bins.append({"lower": i / 10, "upper": (i + 1) / 10, "count": len(subset),
                     "mean_probability": mean(r["pred_probability"] for r in subset) if subset else None,
                     "accuracy": mean(r["pred"] == r["gold"] for r in subset) if subset else None})
    ece = sum(b["count"] * abs(b["mean_probability"] - b["accuracy"]) for b in bins if b["count"]) / len(calibrated) if calibrated else None
    # Rank by probability of the returned label, NOT proprietary confidence.
    ranked = sorted(calibrated, key=lambda r: (-r["pred_probability"], r["id"]))
    coverage = []
    for target in (0.2, 0.4, 0.6, 0.8, 1.0):
        k = min(len(ranked), math.floor(target * n))
        selected = ranked[:k]
        accuracy = mean(r["pred"] == r["gold"] for r in selected) if k else None
        coverage.append({"target_coverage": target, "actual_coverage": k / n if n else None,
                         "accepted": k, "accuracy": accuracy, "risk": 1 - accuracy if k else None})
    return {"n": n, "valid": len(successful), "correct": correct,
            "accuracy": correct / n if n else None, "accuracy_ci95": wilson(correct, n),
            "valid_only_accuracy": correct / len(successful) if successful else None,
            "macro_f1": mean(f1s) if f1s else None, "per_class": per_class, "confusion": matrix,
            "brier_sum": brier, "ece_10_bins": ece, "calibration_n": len(calibrated),
            "reliability_bins": bins, "coverage": coverage}


def question_records(samples, rows, provider):
    lookup = {r["sample_id"]: r for r in rows if r["provider"] == provider and r["phase"] == "measure" and r["round"] == 0}
    grouped = defaultdict(list)
    schemas = {}
    for sample in samples:
        row = lookup.get(sample["id"], {})
        for qid, q in sample["questions"].items():
            group = sample["task"] + "/" + qid
            schemas[group] = q
            answer = row.get("answers", {}).get(qid) if row.get("ok") else None
            typ = q["type"]
            pred = answer["predicted"] if answer else None
            gold = sample["expected"][qid]
            if typ != "score":
                gold, pred = label(gold), label(pred) if pred is not None else None
            grouped[group].append({"id": sample["id"], "gold": gold, "pred": pred,
                                   "probabilities": answer.get("probabilities") if answer else None,
                                   "pred_probability": answer.get("pred_probability") if answer else None})
    return grouped, schemas


def summarize(samples, rows, provider_names, bootstrap=2000, seed=42):
    summary = {"providers": {}, "paired": [], "accuracy_round": 0}
    for provider in provider_names:
        measured = [r for r in rows if r["provider"] == provider and r["phase"] == "measure"]
        valid = [r for r in measured if r["ok"]]
        groups, schemas = question_records(samples, rows, provider)
        tasks = {}
        correct, total = 0, 0
        for group, records in groups.items():
            q = schemas[group]
            if q["type"] == "score":
                errors = [r["pred"] - r["gold"] for r in records if r["pred"] is not None]
                tasks[group] = {"type": "score", "n": len(records), "valid": len(errors),
                                "mae": mean(abs(e) for e in errors) if errors else None,
                                "rmse": math.sqrt(mean(e * e for e in errors)) if errors else None}
            else:
                labels = list(q["criteria"]) if q["type"] == "choice" else ["false", "true"]
                m = classification_metrics(records, labels)
                tasks[group] = {"type": q["type"], **m}
                correct += m["correct"]
                total += m["n"]
        buckets = {}
        for bucket in ("0-1000", "1001-4000", "4001+"):
            rs = [r for r in measured if r["input_chars_bucket"] == bucket]
            vs = [r["latency_ms"] for r in rs if r["ok"]]
            buckets[bucket] = {"requests": len(rs), "valid": len(vs), "p50_ms": percentile(vs, 0.5), "p95_ms": percentile(vs, 0.95)}
        initial = {r["sample_id"]: r for r in measured if r["round"] == 0}
        same, repeat_n = 0, 0
        for r in measured:
            base = initial.get(r["sample_id"])
            if r["round"] > 0 and r["ok"] and base and base["ok"]:
                for qid, a in r["answers"].items():
                    if a["type"] != "score":
                        repeat_n += 1
                        same += a["predicted"] == base["answers"][qid]["predicted"]
        summary["providers"][provider] = {
            "classification_correct": correct, "classification_n": total,
            "classification_accuracy": correct / total if total else None,
            "requests": len(measured), "successful_requests": len(valid),
            "failure_rate": 1 - len(valid) / len(measured) if measured else None,
            "timeout_count": sum(r["error"] == "timeout" for r in measured),
            "p50_ms": percentile([r["latency_ms"] for r in valid], 0.5),
            "p95_ms": percentile([r["latency_ms"] for r in valid], 0.95),
            "all_attempt_p50_ms": percentile([r["latency_ms"] for r in measured], 0.5),
            "all_attempt_p95_ms": percentile([r["latency_ms"] for r in measured], 0.95),
            "errors": dict(Counter(r["error"] for r in measured if not r["ok"])),
            "served_models": sorted({str(r.get("served_model")) for r in valid}),
            "upstream_providers": sorted({str(r.get("upstream_provider")) for r in valid}),
            "repeat_agreement": same / repeat_n if repeat_n else None, "repeat_comparisons": repeat_n,
            "input_length_buckets": buckets, "tasks": tasks,
        }
    for left, right in combinations(provider_names, 2):
        lg, schema = question_records(samples, rows, left)
        rg, _ = question_records(samples, rows, right)
        clusters = defaultdict(list)
        for group in lg:
            if schema[group]["type"] == "score":
                continue
            for a, b in zip(lg[group], rg[group]):
                clusters[a["id"]].append(int(a["pred"] == a["gold"]) - int(b["pred"] == b["gold"]))
        if clusters:
            # Paired cluster bootstrap: resample input records, not repetitions/questions.
            values = [(sum(ds), len(ds)) for ds in clusters.values()]
            observed = sum(v[0] for v in values) / sum(v[1] for v in values)
            rng, draws = random.Random(seed), []
            for _ in range(bootstrap):
                picked = rng.choices(values, k=len(values))
                draws.append(sum(v[0] for v in picked) / sum(v[1] for v in picked))
            summary["paired"].append({"left": left, "right": right, "metric": "classification_accuracy_difference",
                                      "difference": observed, "ci95": [percentile(draws, .025), percentile(draws, .975)],
                                      "unique_sample_clusters": len(values), "bootstrap_draws": bootstrap,
                                      "note": "positive favors left; failures count as wrong; round 0 only; pairwise exploratory CI, no multiplicity correction"})
    return summary
