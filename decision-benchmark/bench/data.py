import hashlib
import json
import math
from pathlib import Path


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_dataset(path):
    samples, ids, schemas = [], set(), {}
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("sample must be an object")
            for key in ("id", "task", "state", "questions", "expected"):
                if key not in item:
                    raise ValueError(f"missing {key}")
            if not isinstance(item["id"], str) or not item["id"] or item["id"] in ids:
                raise ValueError("id must be a unique nonempty string")
            if not isinstance(item["task"], str) or not item["task"]:
                raise ValueError("task must be a nonempty string")
            if not isinstance(item["state"], (str, dict, list)):
                raise ValueError("state must be string/object/array")
            if not isinstance(item["questions"], dict) or not item["questions"]:
                raise ValueError("questions must be a nonempty object")
            if not isinstance(item["expected"], dict) or set(item["questions"]) != set(item["expected"]):
                raise ValueError("expected must contain exactly the question ids")
            for qid, q in item["questions"].items():
                if not isinstance(q, dict):
                    raise ValueError("question must be an object")
                typ, criteria, gold = q.get("type"), q.get("criteria"), item["expected"][qid]
                if typ == "choice":
                    if not isinstance(criteria, dict) or not 1 <= len(criteria) <= 255:
                        raise ValueError("choice needs 1..255 criteria")
                    if not all(isinstance(k, str) and isinstance(v, str) for k, v in criteria.items()):
                        raise ValueError("choice criteria must map strings to strings")
                    if not isinstance(gold, str) or gold not in criteria:
                        raise ValueError("choice expected must be an option key")
                elif typ == "noul":
                    if not isinstance(gold, bool):
                        raise ValueError("noul expected must be JSON true or false")
                elif typ == "score":
                    if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
                        raise ValueError("portable score rubric needs 2..10 levels")
                    if not all(isinstance(c, str) for c in criteria):
                        raise ValueError("score criteria must contain strings")
                    if not numeric(gold) or not 0 <= gold <= len(criteria) - 1:
                        raise ValueError("score expected must be a valid zero-based index/value")
                else:
                    raise ValueError(f"unsupported question type {typ}")
            # One task must retain the exact rubric, instructions and question ids.
            schema = json.dumps(item["questions"], ensure_ascii=False, sort_keys=True)
            if item["task"] in schemas and schemas[item["task"]] != schema:
                raise ValueError("same task has different questions/rubric; use a separate task name")
            schemas[item["task"]] = schema
            ids.add(item["id"])
            samples.append(item)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not samples:
        raise ValueError("dataset is empty")
    return samples


def payload(sample, model):
    # Explicit whitelist: expected labels, ids and metadata NEVER reach the model.
    return {"model": model, "state": sample["state"], "questions": sample["questions"]}


def digest(value):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_answers(body, sample, threshold=0.5):
    if not isinstance(body, dict) or not isinstance(body.get("answers"), dict):
        raise ValueError("response must contain answers object")
    out = {}
    for qid, question in sample["questions"].items():
        a = body["answers"].get(qid)
        typ = question["type"]
        if not isinstance(a, dict) or a.get("type") != typ:
            raise ValueError(f"{qid}: missing answer or incorrect type")
        confidence = a.get("confidence")
        if confidence is not None and (not numeric(confidence) or not 0 <= confidence <= 1):
            raise ValueError(f"{qid}: invalid confidence")
        warnings = []
        if typ == "noul":
            p = a.get("noul")
            if not numeric(p) or not 0 <= p <= 1:
                raise ValueError(f"{qid}: noul must be finite in [0,1]")
            predicted = p >= threshold
            probabilities = {"true": p, "false": 1 - p}
            pred_probability = p if predicted else 1 - p
        else:
            labels = list(question["criteria"]) if typ == "choice" else [str(i) for i in range(len(question["criteria"]))]
            probabilities = a.get("probabilities")
            predicted = a.get(typ)
            if typ == "choice" and (not isinstance(predicted, str) or predicted not in labels):
                raise ValueError(f"{qid}: choice outside criteria")
            if typ == "score" and (not numeric(predicted) or not 0 <= predicted <= len(labels) - 1):
                raise ValueError(f"{qid}: score outside rubric")
            if probabilities is not None:
                if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
                    raise ValueError(f"{qid}: incomplete probability distribution")
                if not all(numeric(p) and 0 <= p <= 1 for p in probabilities.values()):
                    raise ValueError(f"{qid}: invalid probability")
                total = sum(probabilities.values())
                if total <= 0 or abs(total - 1) > 0.0500001:
                    raise ValueError(f"{qid}: probability sum outside 1 +/- 0.05")
                if abs(total - 1) > 1e-9:
                    warnings.append(f"probability_sum={total:.8f}; normalized for metrics")
                probabilities = {k: v / total for k, v in probabilities.items()}
            else:
                warnings.append("probabilities missing; calibration metrics unavailable")
            pred_probability = probabilities[predicted] if typ == "choice" and probabilities else None
        out[qid] = {"type": typ, "predicted": predicted, "probabilities": probabilities,
                    "confidence": confidence, "pred_probability": pred_probability, "warnings": warnings}
    return out
