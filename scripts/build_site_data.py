#!/usr/bin/env python3
"""Rebuild site/data/benchmark.json from a real benchmark run.

Looks for the newest complete, non-mock run under decision-benchmark/results/.
Those run directories are gitignored (large, fully reproducible), so in CI this
script normally finds nothing and keeps the committed JSON as-is. Run it locally
after a new benchmark to refresh the site.

Never writes credentials: only derived metrics and the public PubMedQA text.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "decision-benchmark" / "results"
OUT = ROOT / "site" / "data" / "benchmark.json"
TASK = "pubmedqa/answer"
CN = {
    "bailian": "阿里云百炼",
    "microsoft": "微软 Decision-1",
    "qwen": "Qwen3.5-9B",
    "jev": "TypeSafe Jev 1.13",
}
SECRET = re.compile(
    r"sk-or-v1-[A-Za-z0-9]{10}|sk-[A-Za-z0-9]{25}|github_pat_[A-Za-z0-9]{10}"
    r"|Bearer\s+[A-Za-z0-9]{10}|https://[a-z0-9]{6,}\.cn-beijing\.maas",
    re.I,
)


def pick_run():
    """Newest run that is complete, real, and carries the PubMedQA task."""
    if not RESULTS.is_dir():
        return None
    for d in sorted(RESULTS.iterdir(), key=lambda p: p.name, reverse=True):
        meta_path = d / "run.json"
        summary_path = d / "summary.json"
        if not (meta_path.is_file() and summary_path.is_file()):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if meta.get("status") != "complete" or meta.get("mock"):
            continue
        providers = summary.get("providers", {})
        if set(CN) <= set(providers) and all(TASK in providers[n]["tasks"] for n in CN):
            return d, meta, summary
    return None


def build(run_dir, meta, summary):
    names = [n for n in ("bailian", "microsoft", "qwen", "jev")]
    samples = [json.loads(l) for l in (run_dir / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [json.loads(l) for l in (run_dir / "requests.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = {s["id"]: s["expected"]["answer"] for s in samples}
    first = {
        n: {r["sample_id"]: r for r in rows
            if r["provider"] == n and r["phase"] == "measure" and r["round"] == 0}
        for n in names
    }

    prov = {}
    for n in names:
        p = summary["providers"][n]
        t = p["tasks"][TASK]
        prov[n] = {
            "name": CN[n], "accuracy": t["accuracy"], "correct": t["correct"], "n": t["n"],
            "ci": t["accuracy_ci95"], "macro_f1": t["macro_f1"],
            "brier": t["brier_sum"], "ece": t["ece_10_bins"], "calib_n": t["calibration_n"],
            "p50": p["p50_ms"], "p95": p["p95_ms"],
            "requests": p["requests"], "ok": p["successful_requests"],
            "repeat": p["repeat_agreement"],
            "served": (p["served_models"] or [""])[0],
            "upstream": (p["upstream_providers"] or [""])[0],
            "per_class": {k: {"f1": v["f1"], "precision": v["precision"],
                              "recall": v["recall"], "support": v["support"]}
                          for k, v in t["per_class"].items()},
            "confusion": t["confusion"],
            "coverage": [{"cov": c["actual_coverage"], "acc": c["accuracy"]}
                         for c in t["coverage"] if c["accuracy"] is not None],
            "errors": p["errors"],
        }

    items = []
    for s in samples:
        rec = {"id": s["id"], "q": s["state"]["research_question"],
               "ctx": " ".join(s["state"]["context"])[:1200],
               "gold": gold[s["id"]], "p": {}}
        for n in names:
            r = first[n].get(s["id"])
            if r and r["ok"]:
                a = r["answers"]["answer"]
                rec["p"][n] = {"pred": a["predicted"], "prob": a.get("pred_probability"),
                               "ms": round(r["latency_ms"], 1)}
            else:
                rec["p"][n] = {"pred": None, "prob": None, "ms": None}
        items.append(rec)

    return {
        "meta": {
            "samples": len(samples), "rounds": meta["rounds"],
            "total_requests": sum(summary["providers"][n]["requests"] for n in names),
            "ok_requests": sum(summary["providers"][n]["successful_requests"] for n in names),
            "wall_min": round(meta["wall_seconds_including_warmup"] / 60, 1),
            "seed": meta["seed"], "created": meta["created_at"], "run_id": run_dir.name,
            "bootstrap": meta["bootstrap_draws"], "dataset_sha256": meta["dataset_sha256"],
        },
        "order": names, "providers": prov,
        "paired": summary["paired"], "items": items,
    }


def main():
    found = pick_run()
    if not found:
        if OUT.is_file():
            print(f"no local benchmark run found; keeping committed {OUT.relative_to(ROOT)}")
            return 0
        print("ERROR: no benchmark run and no committed site data", file=sys.stderr)
        return 1

    run_dir, meta, summary = found
    data = build(run_dir, meta, summary)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    hit = SECRET.search(text)
    if hit:
        print(f"ERROR: refusing to write site data containing a secret-like value: "
              f"{hit.group(0)[:12]}...", file=sys.stderr)
        return 2

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} from run {run_dir.name} "
          f"({len(data['items'])} items, {len(text):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
