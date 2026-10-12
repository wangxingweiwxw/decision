import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from bench.providers import get_providers, Client
from bench.data import payload
from bench.cli import main
from bench.metrics import summarize
from test_benchmark import SAMPLE, result, response

ROOT = Path(__file__).resolve().parent.parent


class JevProviderTests(unittest.TestCase):
    def test_jev_provider_shape(self):
        p = get_providers(["jev"], mock=True)[0]
        self.assertEqual(p.name, "jev")
        self.assertEqual(p.protocol, "decision")
        # Jev routes through OpenRouter's Decisions API, same surface as microsoft.
        self.assertEqual(p.url, "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(p.model, "typesafe/jev-1.13")
        # Jev shares the native decision payload with bailian/microsoft (no chat schema).
        self.assertEqual(set(payload(SAMPLE, p.model)), {"model", "state", "questions"})

    def test_jev_requires_openrouter_key_for_real_run(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                get_providers(["jev"], mock=False)

    def test_jev_model_and_provider_overrides(self):
        with patch.dict(os.environ, {"JEV_MODEL": "~typesafe/jev-latest",
                                     "JEV_PROVIDER": "typesafe"}, clear=False):
            p = get_providers(["jev"], mock=True)[0]
            self.assertEqual(p.model, "~typesafe/jev-latest")
            self.assertEqual(p.extra["provider"], {"only": ["typesafe"], "allow_fallbacks": False})

    def test_jev_transport_sets_openrouter_title_header(self):
        p = get_providers(["jev"], mock=True)[0]
        c = Client(p)
        self.addCleanup(c.close)
        # Jev is routed through OpenRouter; the OpenRouter ranking header must be present.
        self.assertEqual(c.session.headers.get("X-Title"), "Decision Benchmark")

    def test_jev_native_probabilities_preserved(self):
        p = get_providers(["jev"], mock=True)[0]
        c = Client(p)
        self.addCleanup(c.close)
        body = {**response("a", {"a": 0.9, "b": 0.1}), "model": p.model, "provider": "TypeSafe"}
        http = Mock(status_code=200, content=b"{}")
        http.json.return_value = body
        with patch.object(c.session, "post", return_value=http) as post:
            row = c.call(SAMPLE)
        self.assertTrue(row["ok"])
        self.assertEqual(row["probability_source"], "native_decision_api")
        self.assertEqual(row["request_payload"], post.call_args.kwargs["json"])
        # Native decision API keeps calibrated probabilities, unlike the Qwen chat baseline.
        self.assertIsNotNone(row["answers"]["q"]["probabilities"])


class FourWayTests(unittest.TestCase):
    def test_six_pairs_for_four_providers(self):
        names = ["bailian", "microsoft", "qwen", "jev"]
        samples = [{**SAMPLE, "id": str(i)} for i in range(4)]
        rows = [result(s, p, "b" if p == "qwen" else "a") for s in samples for p in names]
        out = summarize(samples, rows, names, 30)
        self.assertEqual(len(out["paired"]), 6)  # C(4,2)
        pairs = {(x["left"], x["right"]) for x in out["paired"]}
        self.assertIn(("bailian", "jev"), pairs)
        self.assertIn(("microsoft", "jev"), pairs)
        self.assertIn(("qwen", "jev"), pairs)

    def test_four_way_mock_full_run(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = Path(d) / "run"
            code = main(["run", "--providers", "bailian,microsoft,qwen,jev", "--mock",
                         "--limit", "2", "--rounds", "2", "--warmup", "1", "--delay", "0",
                         "--bootstrap", "20", "--output", str(out)])
            self.assertEqual(code, 0)
            meta = json.loads((out / "run.json").read_text(encoding="utf-8"))
            self.assertEqual([p["name"] for p in meta["providers"]], ["bailian", "microsoft", "qwen", "jev"])
            s = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(s["paired"]), 6)
            self.assertEqual(s["providers"]["jev"]["requests"], 4)
            # Jev returns native probabilities in mock, so calibration metrics are available.
            for task in s["providers"]["jev"]["tasks"].values():
                self.assertIsNotNone(task["brier_sum"])

    def test_report_names_jev(self):
        from bench.report import display
        self.assertEqual(display("jev"), "TypeSafe Jev")


if __name__ == "__main__":
    unittest.main()
