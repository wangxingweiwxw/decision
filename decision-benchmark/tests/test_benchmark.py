import contextlib
import copy
import io
import json
import math
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import requests

from bench.data import load_dataset, normalize_answers, payload
from bench.metrics import classification_metrics, summarize, percentile, wilson
from bench.providers import Client, Provider, redact
from bench.cli import main

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = {"id": "1", "task": "t", "state": "evidence", "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}, "expected": {"q": "a"}}


def response(choice="a", probs=None):
    return {"model": "test", "answers": {"q": {"type": "choice", "choice": choice, "probabilities": probs if probs is not None else {"a": .8, "b": .2}, "confidence": .7}}, "usage": {"input_tokens": 2}}


def result(sample, provider, prediction="a", round_no=0, ok=True):
    return {"sample_id": sample["id"], "provider": provider, "phase": "measure", "round": round_no, "ok": ok,
            "error": None if ok else "timeout", "latency_ms": 10, "input_chars_bucket": "0-1000",
            "answers": normalize_answers(response(prediction), sample) if ok else {}}


class DataTests(unittest.TestCase):
    def test_explicit_dotenv_override(self):
        import os
        from bench.providers import load_env
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / '.env'
            p.write_text('EXAMPLE_TEST_KEY=new-value\n', encoding='utf-8')
            with patch.dict(os.environ, {'EXAMPLE_TEST_KEY': 'old-value'}):
                load_env(p)
                self.assertEqual(os.environ['EXAMPLE_TEST_KEY'], 'old-value')
                load_env(p, override=True)
                self.assertEqual(os.environ['EXAMPLE_TEST_KEY'], 'new-value')

    def test_no_labels_in_payload(self):
        s = {**SAMPLE, "metadata": {"answer": "do-not-send"}}
        self.assertEqual(set(payload(s, "m")), {"model", "state", "questions"})
        self.assertNotIn("do-not-send", json.dumps(payload(s, "m")))

    def test_duplicates_and_rubric_drift(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'd.jsonl'
            p.write_text('\n'.join(json.dumps(SAMPLE) for _ in range(2)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unique'):
                load_dataset(p)
            changed = copy.deepcopy(SAMPLE)
            changed['id'] = '2'
            changed['questions']['q']['criteria']['b'] = 'different meaning'
            p.write_text('\n'.join(map(json.dumps, [SAMPLE, changed])), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'rubric'):
                load_dataset(p)

    def test_bundled_data(self):
        self.assertEqual(len(load_dataset(ROOT / 'data/science_demo.jsonl')), 12)
        self.assertEqual(len(load_dataset(ROOT / 'data/mixed_demo.jsonl')), 4)

    def test_rounding_normalization(self):
        a = normalize_answers(response(probs={'a': .95, 'b': .04}), SAMPLE)['q']
        self.assertAlmostEqual(sum(a['probabilities'].values()), 1)
        self.assertTrue(a['warnings'])

    def test_bad_probabilities(self):
        for probs in ({'a': .4, 'b': .4}, {'a': float('nan'), 'b': .2}, {'a': 1.1, 'b': -.1}, {'a': 1}):
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                normalize_answers(response(probs=probs), SAMPLE)

    def test_invalid_choice_and_missing_answer(self):
        with self.assertRaises(ValueError):
            normalize_answers(response('c'), SAMPLE)
        with self.assertRaises(ValueError):
            normalize_answers({'answers': {}}, SAMPLE)

    def test_noul_threshold_and_score(self):
        s = load_dataset(ROOT / 'data/mixed_demo.jsonl')[0]
        c = Client(Provider('microsoft', '', 'm', ''), mock=True)
        body = c.mock_body(s)
        body['answers']['converged']['noul'] = .5
        a = normalize_answers(body, s)
        self.assertIs(a['converged']['predicted'], True)
        body['answers']['completeness']['score'] = 99
        with self.assertRaises(ValueError):
            normalize_answers(body, s)
        c.close()


class MetricTests(unittest.TestCase):
    def test_known_metrics_with_failure(self):
        records = [
            {'id': '1', 'gold': 'a', 'pred': 'a', 'probabilities': {'a': .8, 'b': .2}, 'pred_probability': .8},
            {'id': '2', 'gold': 'b', 'pred': 'a', 'probabilities': {'a': .6, 'b': .4}, 'pred_probability': .6},
            {'id': '3', 'gold': 'b', 'pred': None, 'probabilities': None, 'pred_probability': None},
        ]
        m = classification_metrics(records, ['a', 'b'])
        self.assertAlmostEqual(m['accuracy'], 1/3)
        self.assertAlmostEqual(m['macro_f1'], 1/3)
        self.assertAlmostEqual(m['brier_sum'], .4)
        self.assertAlmostEqual(m['ece_10_bins'], .4)
        self.assertEqual(m['confusion']['b']['<ERROR>'], 1)
        self.assertAlmostEqual(m['coverage'][-1]['actual_coverage'], 2/3)

    def test_repeats_do_not_inflate_accuracy(self):
        rows = [result(SAMPLE, 'a'), result(SAMPLE, 'b', 'b')]
        rows += [result(SAMPLE, 'a', 'b', 1), result(SAMPLE, 'b', 'a', 1)]
        out = summarize([SAMPLE], rows, ['a', 'b'], 100)
        self.assertEqual(out['providers']['a']['classification_n'], 1)
        self.assertEqual(out['providers']['a']['classification_accuracy'], 1)
        self.assertEqual(out['paired'][0]['difference'], 1)
        self.assertEqual(out['paired'][0]['unique_sample_clusters'], 1)
        self.assertEqual(out['providers']['a']['repeat_agreement'], 0)

    def test_warmups_excluded_and_errors_counted(self):
        warm = {**result(SAMPLE, 'a'), 'phase': 'warmup', 'latency_ms': 9999}
        out = summarize([SAMPLE], [warm, result(SAMPLE, 'a', ok=False)], ['a'], 20)
        p = out['providers']['a']
        self.assertEqual(p['classification_accuracy'], 0)
        self.assertEqual(p['requests'], 1)
        self.assertIsNone(p['p50_ms'])
        self.assertEqual(p['failure_rate'], 1)

    def test_cluster_bootstrap_same_models(self):
        samples = [{**SAMPLE, 'id': str(i)} for i in range(5)]
        rows = [result(s, p) for s in samples for p in ['a', 'b']]
        pair = summarize(samples, rows, ['a', 'b'], 200)['paired'][0]
        self.assertEqual(pair['ci95'], [0, 0])

    def test_percentiles_and_wilson(self):
        self.assertEqual(percentile([1, 2, 3], .5), 2)
        self.assertAlmostEqual(percentile([0, 100], .95), 95)
        self.assertTrue(0 < wilson(5, 10)[0] < .5 < wilson(5, 10)[1] < 1)


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    captured = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        type(self).captured.append(body)
        status = 200
        if self.path == '/unauthorized':
            status, raw = 401, b'{"error":"sk-secret-test-123456"}'
        elif self.path == '/invalid':
            raw = b'not json'
        elif self.path == '/missing':
            raw = b'{"answers":{}}'
        else:
            raw = json.dumps(response()).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def client(self, path):
        c = Client(Provider('microsoft', f'http://127.0.0.1:{self.server.server_port}{path}', 'm', 'sk-secret-test-123456'))
        c.session.trust_env = False
        self.addCleanup(c.close)
        return c

    def test_real_http_transport_and_payload(self):
        r = self.client('/ok').call(SAMPLE)
        self.assertTrue(r['ok'])
        self.assertGreater(r['latency_ms'], 0)
        self.assertEqual(set(Handler.captured[-1]), {'model', 'state', 'questions'})

    def test_http_error_redacts_secret(self):
        r = self.client('/unauthorized').call(SAMPLE)
        self.assertFalse(r['ok'])
        self.assertEqual(r['http_status'], 401)
        self.assertNotIn('sk-secret-test-123456', json.dumps(r))

    def test_invalid_json_and_missing_answer(self):
        self.assertFalse(self.client('/invalid').call(SAMPLE)['ok'])
        self.assertFalse(self.client('/missing').call(SAMPLE)['ok'])

    def test_timeout_recorded_not_dropped(self):
        c = self.client('/ok')
        with patch.object(c.session, 'post', side_effect=requests.Timeout()):
            r = c.call(SAMPLE)
        self.assertFalse(r['ok'])
        self.assertEqual(r['error'], 'timeout')
        self.assertGreaterEqual(r['latency_ms'], 0)


class CLITests(unittest.TestCase):
    def test_mock_end_to_end_concurrent(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            out = Path(d) / 'run'
            code = main(['run', '--mock', '--dataset', str(ROOT / 'data/mixed_demo.jsonl'), '--rounds', '2', '--warmup', '1', '--delay', '0', '--concurrency', '2', '--bootstrap', '30', '--output', str(out)])
            self.assertEqual(code, 0)
            self.assertTrue((out / 'report.html').exists())
            self.assertIn('模拟', (out / 'report.html').read_text(encoding='utf-8'))
            rows = [json.loads(x) for x in (out / 'requests.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len([r for r in rows if r['phase'] == 'measure']), 16)
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['providers']['bailian']['classification_n'], 8)
            self.assertEqual(main(['report', str(out)]), 0)

    def test_pubmedqa_import_no_answer_leak_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            source, out = Path(d) / 'original.json', Path(d) / 'prepared'
            original = {str(i): {'QUESTION': 'Question?', 'CONTEXTS': ['Evidence'], 'LONG_ANSWER': 'SECRET-ANSWER', 'final_decision': ['yes','no','maybe'][i % 3]} for i in range(30)}
            source.write_text(json.dumps(original), encoding='utf-8')
            self.assertEqual(main(['prepare-pubmedqa', '--source', str(source), '--output-dir', str(out)]), 0)
            dev, test = load_dataset(out / 'dev.jsonl'), load_dataset(out / 'test.jsonl')
            self.assertEqual(len(dev) + len(test), 30)
            self.assertFalse({s['id'] for s in dev} & {s['id'] for s in test})
            self.assertNotIn('SECRET-ANSWER', json.dumps(dev + test))


if __name__ == '__main__':
    unittest.main()
