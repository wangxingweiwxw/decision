import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import random
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from .data import load_dataset, digest
from .metrics import summarize
from .providers import load_env, get_providers, Client
from .report import generate_report

ROOT = Path(__file__).resolve().parent.parent


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def decorate(row, sample, phase, round_no, worker):
    size = len(json.dumps({"state": sample["state"], "questions": sample["questions"]}, ensure_ascii=False))
    return {**row, "phase": phase, "round": round_no, "worker": worker,
            "input_chars": size, "input_chars_bucket": "0-1000" if size <= 1000 else "1001-4000" if size <= 4000 else "4001+"}


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return value


def selected_samples(args):
    samples = load_dataset(args.dataset)
    # Uniform seeded sample, never choose only easy/first rows for --limit.
    if args.limit and args.limit < len(samples):
        samples = random.Random(args.seed).sample(samples, args.limit)
    return samples


def run(args):
    if not 0 <= args.threshold <= 1 or args.timeout <= 0 or args.delay < 0:
        raise ValueError("threshold must be in [0,1]; timeout > 0; delay >= 0")
    samples = selected_samples(args)
    names = args.providers.split(",")
    if len(names) != len(set(names)) or not names:
        raise ValueError("providers must be unique")
    providers = get_providers(names, args.mock)
    if any(p.protocol == "chat_choice" for p in providers) and any(
            q["type"] != "choice" for sample in samples for q in sample["questions"].values()):
        raise ValueError("Qwen baseline supports choice-only datasets; no API calls made")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = Path(args.output) if args.output else ROOT / "results" / stamp
    directory.mkdir(parents=True, exist_ok=False)  # Never mix or overwrite prior runs.
    meta = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "running", "mock": args.mock,
            "dataset_path": str(Path(args.dataset).resolve()), "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
            "selected_samples_sha256": digest(samples), "samples": len(samples), "rounds": args.rounds,
            "warmup_per_worker_per_provider": args.warmup, "concurrency": args.concurrency,
            "timeout_seconds": args.timeout, "delay_between_calls_seconds": args.delay, "seed": args.seed,
            "noul_threshold": args.threshold, "bootstrap_draws": args.bootstrap,
            "providers": [p.public() for p in providers], "python": sys.version.split()[0],
            "platform": platform.platform(), "requests_version": requests.__version__,
            "timing": "perf_counter around HTTP request through full body receipt; excludes JSON parsing; no retries",
            "scheduling": "sample groups randomized per round; provider order randomized per group; per-worker persistent sessions",
            "accuracy": "round 0 only; failures incorrect; paired CI clustered by sample id",
            "limitations": ["OpenRouter gateway and network are included; not isolated model computation",
                            "Server latency fields are recorded but not compared",
                            "Repeated prompts may be cached by providers; no client cache",
                            "Per-call requests timeout is connect/read inactivity, not a strict wall-clock deadline",
                            "Score rubrics use 0-based ordinal values, not physical quantities"]}
    write_json(directory / "run.json", meta)
    with (directory / "dataset.jsonl").open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(("[MOCK 模拟] " if args.mock else "[真实 API] ") + f"{len(samples)} 样本 × {args.rounds} 轮 × {len(providers)} 服务")
    max_calls = (len(samples) * args.rounds + min(args.concurrency, len(samples)) * args.warmup) * len(providers)
    print(f"最多约 {max_calls} 次请求（含暖机），无自动重试。输出: {directory}")
    rows, clients, lock = [], [], threading.Lock()
    local = threading.local()
    rng = random.Random(args.seed)
    start_run = time.perf_counter()
    completed, total = 0, len(samples) * args.rounds
    interrupted = False

    def work(sample, order, round_no):
        batch = []
        if not hasattr(local, "clients"):
            local.clients = {}
        for idx in order:
            provider = providers[idx]
            if provider.name not in local.clients:
                c = Client(provider, args.timeout, args.threshold, args.mock)
                local.clients[provider.name] = c
                with lock:
                    clients.append(c)
                for _ in range(args.warmup):
                    row = decorate(c.call(sample), sample, "warmup", -1, threading.current_thread().name)
                    batch.append(row)
                    if row["http_status"] in (401, 403, 404, 402):
                        return batch, f"{provider.name}: HTTP {row['http_status']}；请先运行 check 并检查密钥、余额、模型权限。"
                    if args.delay:
                        time.sleep(args.delay)
            c = local.clients[provider.name]
            row = decorate(c.call(sample), sample, "measure", round_no, threading.current_thread().name)
            batch.append(row)
            if row["http_status"] in (401, 403, 404, 402):
                return batch, f"{provider.name}: HTTP {row['http_status']}；已停止继续提交任务。"
            if args.delay:
                time.sleep(args.delay)
        return batch, None

    fatal = None
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency)
    output = (directory / "requests.jsonl").open("w", encoding="utf-8")

    def save(batch):
        rows.extend(batch)
        for row in batch:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()

    pending = set()
    try:
        for round_no in range(args.rounds):
            order_samples = list(samples)
            rng.shuffle(order_samples)
            jobs = []
            for sample in order_samples:
                order = list(range(len(providers)))
                rng.shuffle(order)
                jobs.append((sample, order, round_no))
            iterator = iter(jobs)
            # Bounded queue: at most concurrency sample-pairs in flight.
            for _ in range(min(args.concurrency, len(jobs))):
                pending.add(pool.submit(work, *next(iterator)))
            while pending:
                done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    batch, reason = future.result()
                    save(batch)
                    completed += 1
                    fatal = fatal or reason
                    if completed % 10 == 0 or completed == total or reason:
                        print(f"完成 {completed}/{total} 个样本轮次" + (f"；{reason}" if reason else ""), flush=True)
                if not fatal:
                    for _ in done:
                        job = next(iterator, None)
                        if job is not None:
                            pending.add(pool.submit(work, *job))
            if fatal:
                break
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到停止请求，保存已完成结果；等待已发出的 HTTP 请求结束。")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for future in pending:
            if not future.cancelled():
                try:
                    batch, reason = future.result()
                    save(batch)
                    fatal = fatal or reason
                except Exception:
                    pass
        output.close()
        for c in clients:
            c.close()
    measured = [r for r in rows if r["phase"] == "measure"]
    meta.update({"status": "interrupted" if interrupted else "failed" if fatal else "complete",
                 "fatal_error": fatal, "finished_at": datetime.now(timezone.utc).isoformat(),
                 "wall_seconds_including_warmup": time.perf_counter() - start_run,
                 "recorded_requests_including_warmup": len(rows)})
    write_json(directory / "run.json", meta)
    summary = summarize(samples, rows, names, args.bootstrap, args.seed)
    if meta["status"] != "complete":
        summary["paired"] = []
    generate_report(directory, meta, samples, rows, summary)
    print(f"报告: {directory / 'report.html'}")
    for name, m in summary["providers"].items():
        a = m["classification_accuracy"]
        atext = f"{a:.2%}" if a is not None else "N/A"
        print(f"{name}: accuracy={atext}, valid requests={m['successful_requests']}/{m['requests']}, p50={m['p50_ms']}, p95={m['p95_ms']}")
    return 130 if interrupted else 2 if fatal or any(not r["ok"] for r in measured) else 0


def check(args):
    names = args.providers.split(",")
    providers = get_providers(names, args.mock)
    sample = load_dataset(ROOT / "data" / "science_demo.jsonl")[0]
    failed = False
    for p in providers:
        client = Client(p, timeout=30, mock=args.mock)
        try:
            r = client.call(sample)
            print(json.dumps({"provider": p.name, "ok": r["ok"], "http_status": r["http_status"],
                              "served_model": r.get("served_model"), "latency_ms": r["latency_ms"],
                              "error": r["error"]}, ensure_ascii=False))
            failed |= not r["ok"]
        finally:
            client.close()
    return 2 if failed else 0


def report(args):
    directory = Path(args.directory)
    meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    samples = load_dataset(directory / "dataset.jsonl")
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = summarize(samples, rows, [p["name"] for p in meta["providers"]], meta["bootstrap_draws"], meta["seed"])
    if meta["status"] != "complete":
        summary["paired"] = []
    generate_report(directory, meta, samples, rows, summary)
    print(directory / "report.html")
    return 0


def prepare_pubmedqa(args):
    source_url = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
    if args.source:
        raw = Path(args.source).read_bytes()
        source_name = str(Path(args.source).resolve())
    else:
        print("下载 PubMedQA 作者仓库的 PQA-L 数据（不调用模型）…")
        response = requests.get(source_url, timeout=60)
        response.raise_for_status()
        raw, source_name = response.content, source_url
    obj = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError("Expected original PubMedQA ori_pqal.json object keyed by PMID")
    questions = {"answer": {"type": "choice", "instructions": "Answer the research question using only the supplied abstract context. Choose maybe when the evidence is inconclusive.",
                             "criteria": {"yes": "The context supports an affirmative answer.", "no": "The context supports a negative answer.", "maybe": "The context is inconclusive or does not establish a yes/no answer."}}}
    samples = []
    for pmid, obj_item in obj.items():
        if not isinstance(obj_item, dict) or obj_item.get("final_decision") not in ("yes", "no", "maybe"):
            raise ValueError(f"Invalid PubMedQA item {pmid}")
        context, question = obj_item.get("CONTEXTS"), obj_item.get("QUESTION")
        if not isinstance(context, list) or not all(isinstance(v, str) for v in context) or not isinstance(question, str):
            raise ValueError(f"Invalid question/context at PMID {pmid}")
        samples.append({"id": str(pmid), "task": "pubmedqa", "state": {"research_question": question, "context": context},
                        "questions": questions, "expected": {"answer": obj_item["final_decision"]}})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)
    dev, test = [], []
    for gold in ("yes", "no", "maybe"):
        subset = [s for s in samples if s["expected"]["answer"] == gold]
        rng.shuffle(subset)
        n = int(len(subset) * args.dev_fraction)
        dev.extend(subset[:n])
        test.extend(subset[n:])
    for split, records in (("dev", dev), ("test", test)):
        rng.shuffle(records)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    write_json(out / "provenance.json", {"source": source_name, "source_sha256": hashlib.sha256(raw).hexdigest(),
               "retrieved_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "dev": len(dev), "test": len(test),
               "split": "Custom seeded stratified split, NOT the official leaderboard split",
               "excluded_fields": ["LONG_ANSWER", "final_decision (not sent to API)"],
               "citation": "Jin et al., PubMedQA, EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1259/",
               "license_note": "Use subject to upstream dataset terms; inspect original repository before redistribution."})
    print(f"已生成 dev={len(dev)}, test={len(test)}，目录 {out}。这是自定义划分，不可当作官方榜单成绩。")
    return 0


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="阿里云百炼 vs 微软 Decision-1 vs Qwen3.5-9B vs TypeSafe Jev：准确率、概率与端到端延迟基准")
    parser.add_argument("--env-file", default=str(ROOT / ".env"), help="dotenv file (default: project/.env)")
    parser.add_argument("--env-override", action="store_true", help="prefer nonempty .env values over inherited environment variables")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="run paired benchmark (real API unless --mock)")
    run_p.add_argument("--dataset", default=str(ROOT / "data" / "science_demo.jsonl"))
    run_p.add_argument("--providers", default="bailian,microsoft", help="comma-separated: bailian,microsoft,qwen,jev")
    run_p.add_argument("--mock", action="store_true", help="offline fake responses; never evidence of model quality")
    run_p.add_argument("--limit", type=positive)
    run_p.add_argument("--rounds", type=positive, default=3)
    run_p.add_argument("--warmup", type=nonnegative, default=2, help="per worker per provider")
    run_p.add_argument("--concurrency", type=positive, default=1, help="concurrent sample pairs; each provider <= this many calls")
    run_p.add_argument("--timeout", type=float, default=60)
    run_p.add_argument("--delay", type=float, default=0.1, help="pause after each request, outside latency timer")
    run_p.add_argument("--threshold", type=float, default=0.5, help="fixed noul classification threshold")
    run_p.add_argument("--seed", type=int, default=42)
    run_p.add_argument("--bootstrap", type=positive, default=2000)
    run_p.add_argument("--output", help="NEW directory, must not exist")
    check_p = sub.add_parser("check", help="send ONE real request per provider, unless --mock")
    check_p.add_argument("--providers", default="bailian,microsoft")
    check_p.add_argument("--mock", action="store_true")
    validate_p = sub.add_parser("validate", help="validate JSONL dataset locally, no network")
    validate_p.add_argument("dataset")
    report_p = sub.add_parser("report", help="rebuild reports from saved results, no API calls")
    report_p.add_argument("directory")
    prep = sub.add_parser("prepare-pubmedqa", help="download/import PubMedQA and create disjoint dev/test JSONL")
    prep.add_argument("--source", help="local ori_pqal.json; omit to download from authors' repository")
    prep.add_argument("--output-dir", default=str(ROOT / "data" / "pubmedqa"))
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--dev-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)
    try:
        load_env(args.env_file, override=args.env_override)
        if args.command == "run":
            return run(args)
        if args.command == "check":
            return check(args)
        if args.command == "report":
            return report(args)
        if args.command == "prepare-pubmedqa":
            if not 0 < args.dev_fraction < 1:
                raise ValueError("dev-fraction must be between 0 and 1")
            return prepare_pubmedqa(args)
        if args.command == "validate":
            samples = load_dataset(args.dataset)
            print(f"有效：{len(samples)} 条输入，{sum(len(s['questions']) for s in samples)} 个问题。")
            return 0
    except requests.RequestException as exc:
        print(f"下载失败：{type(exc).__name__}。可手动下载 ori_pqal.json 后通过 --source 导入。", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
