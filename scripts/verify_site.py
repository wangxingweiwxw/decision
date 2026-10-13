#!/usr/bin/env python3
"""Gate the Pages deploy: fail the build rather than publish a broken site.

Checks the static bundle in site/ for required files, data integrity,
internal consistency of the headline numbers, and absence of credentials.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
DATA = SITE / "data" / "benchmark.json"
REQUIRED = ["index.html", "assets/style.css", "assets/app.js", "data/benchmark.json"]
SECRET = re.compile(
    r"sk-or-v1-[A-Za-z0-9]{10}|sk-[A-Za-z0-9]{25}|github_pat_[A-Za-z0-9]{10}"
    r"|Bearer\s+[A-Za-z0-9]{10}|DASHSCOPE_API_KEY\s*=\s*\S"
    r"|https://[a-z0-9]{6,}\.cn-beijing\.maas",
    re.I,
)

errors = []
notes = []


def check(cond, msg):
    if cond:
        notes.append(f"ok   {msg}")
    else:
        errors.append(msg)


# --- bundle completeness
for rel in REQUIRED:
    check((SITE / rel).is_file(), f"present: {rel}")

if not DATA.is_file():
    print("FATAL: site/data/benchmark.json missing", file=sys.stderr)
    raise SystemExit(1)

raw = DATA.read_text(encoding="utf-8")

# --- no credentials anywhere in the shipped bundle
for f in sorted(SITE.rglob("*")):
    if not f.is_file() or f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".ico", ".svg"}:
        continue
    try:
        body = f.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    hit = SECRET.search(body)
    check(hit is None, f"no credential pattern in {f.relative_to(SITE)}"
                       + (f" (found {hit.group(0)[:12]}...)" if hit else ""))

# --- data integrity
try:
    d = json.loads(raw)
except ValueError as e:
    print(f"FATAL: benchmark.json is not valid JSON: {e}", file=sys.stderr)
    raise SystemExit(1)

for key in ("meta", "order", "providers", "paired", "items"):
    check(key in d, f"data has '{key}'")

m, order, prov = d["meta"], d["order"], d["providers"]
check(len(order) == 4, f"four providers declared (got {len(order)})")
check(all(n in prov for n in order), "every declared provider has a block")
check(len(d["paired"]) == 6, f"six pairwise comparisons (got {len(d['paired'])})")
check(len(d["items"]) == m["samples"],
      f"item count matches meta.samples ({len(d['items'])} vs {m['samples']})")

# headline numbers must be internally consistent
tot = sum(prov[n]["requests"] for n in order)
ok = sum(prov[n]["ok"] for n in order)
check(tot == m["total_requests"], f"provider requests sum to meta.total_requests ({tot})")
check(ok == m["ok_requests"], f"provider successes sum to meta.ok_requests ({ok})")
check(ok <= tot, "successes never exceed total requests")

for n in order:
    p = prov[n]
    check(0 <= p["accuracy"] <= 1, f"{n}: accuracy within [0,1]")
    check(p["correct"] <= p["n"], f"{n}: correct <= n")
    check(abs(p["accuracy"] - p["correct"] / p["n"]) < 1e-9,
          f"{n}: accuracy equals correct/n")
    check(p["ci"][0] <= p["accuracy"] <= p["ci"][1], f"{n}: accuracy inside its Wilson interval")
    check(p["p50"] <= p["p95"], f"{n}: p50 <= p95")
    # every item must carry a slot for every provider
    check(all(n in it["p"] for it in d["items"]), f"{n}: present on all items")

# a provider with no native probabilities must not fake them
for n in order:
    if prov[n]["calib_n"] == 0:
        bad = [it["id"] for it in d["items"] if it["p"][n]["prob"] is not None]
        check(not bad, f"{n}: no fabricated probabilities (calib_n=0)")
        check(prov[n]["brier"] is None, f"{n}: Brier reported as unavailable")

# gold labels must never be empty, and preds must be a known label or null
labels = {"yes", "no", "maybe"}
bad_gold = [it["id"] for it in d["items"] if it["gold"] not in labels]
check(not bad_gold, f"all gold labels valid (bad: {bad_gold[:3]})")
bad_pred = [it["id"] for it in d["items"] for n in order
            if it["p"][n]["pred"] not in labels and it["p"][n]["pred"] is not None]
check(not bad_pred, f"all predictions are valid labels or null (bad: {bad_pred[:3]})")

# --- report
for line in notes:
    print(line)
if errors:
    print(f"\n{len(errors)} check(s) FAILED:", file=sys.stderr)
    for e in errors:
        print(f"  fail {e}", file=sys.stderr)
    raise SystemExit(1)
print(f"\nall {len(notes)} checks passed · {m['samples']} items · "
      f"{m['total_requests']:,} requests · run {m['run_id']}")
