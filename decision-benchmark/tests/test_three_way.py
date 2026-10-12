import copy
import json
import tempfile
import unittest
import contextlib
import io
from pathlib import Path
from unittest.mock import patch
from bench.providers import chat_choice_payload, normalize_chat_choices, get_providers, Client
from bench.cli import main
from bench.metrics import summarize
from test_benchmark import SAMPLE, result

ROOT=Path(__file__).resolve().parent.parent

class ThreeWayTests(unittest.TestCase):
    def response(self, content='{"q":"a"}', finish='stop'):
        return {'choices':[{'finish_reason':finish,'message':{'content':content}}]}

    def test_chat_payload_no_gold_leak(self):
        sample={**copy.deepcopy(SAMPLE),'expected':{'q':'SECRET-GOLD'},'metadata':{'hidden':'SECRET-META'}}
        p=chat_choice_payload(sample,'qwen/qwen3.5-9b')
        self.assertNotIn('SECRET-',json.dumps(p))
        self.assertEqual(json.loads(p['messages'][1]['content']),{'state':sample['state'],'questions':sample['questions']})
        self.assertEqual(p['response_format']['json_schema']['schema']['properties']['q']['enum'],['a','b'])

    def test_no_fabricated_probabilities(self):
        a=normalize_chat_choices(self.response(),SAMPLE)['q']
        self.assertEqual(a['predicted'],'a')
        for k in ['probabilities','pred_probability','confidence']: self.assertIsNone(a[k])

    def test_strict_validation(self):
        for content,finish in [('not json','stop'),('{"q":"c"}','stop'),('{"q":"a","extra":"b"}','stop'),('{}','stop'),('{"q":"a"}','length')]:
            with self.subTest(content=content,finish=finish),self.assertRaises(ValueError):
                normalize_chat_choices(self.response(content,finish),SAMPLE)

    def test_transport_adapter_and_recorded_payload(self):
        p=get_providers(['qwen'],mock=True)[0];c=Client(p)
        self.addCleanup(c.close)
        from unittest.mock import Mock
        body={**self.response(),'model':p.model,'provider':'DeepInfra','usage':{'prompt_tokens':99}}
        response=Mock(status_code=200,content=b'{}');response.json.return_value=body
        with patch.object(c.session,'post',return_value=response) as post:
            row=c.call(SAMPLE)
        self.assertTrue(row['ok']);self.assertEqual(row['served_model'],p.model)
        self.assertEqual(row['request_payload'],post.call_args.kwargs['json'])
        self.assertEqual(row['probability_source'],'unavailable')
        self.assertFalse(post.call_args.kwargs['json']['reasoning']['enabled'])

    def test_three_pairs(self):
        samples=[{**SAMPLE,'id':str(i)} for i in range(4)]
        rows=[result(s,p,'b' if p=='qwen' else 'a') for s in samples for p in ['bailian','microsoft','qwen']]
        out=summarize(samples,rows,['bailian','microsoft','qwen'],30)
        self.assertEqual(len(out['paired']),3)
        self.assertEqual([p['difference'] for p in out['paired']],[0,1,1])

    def test_three_way_mock_full_run(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()):
            out=Path(d)/'run'
            self.assertEqual(main(['run','--providers','bailian,microsoft,qwen','--mock','--limit','2','--rounds','2','--warmup','1','--delay','0','--bootstrap','20','--output',str(out)]),0)
            s=json.loads((out/'summary.json').read_text())
            self.assertEqual(len(s['paired']),3)
            self.assertEqual(s['providers']['qwen']['classification_n'],2)
            self.assertEqual(s['providers']['qwen']['requests'],4)
            for task in s['providers']['qwen']['tasks'].values(): self.assertIsNone(task['brier_sum'])

    def test_reject_unsupported_before_network(self):
        with tempfile.TemporaryDirectory() as d,contextlib.redirect_stderr(io.StringIO()),patch.object(Client,'call') as call:
            code=main(['run','--providers','bailian,microsoft,qwen','--mock','--dataset',str(ROOT/'data/mixed_demo.jsonl'),'--output',str(Path(d)/'out')])
            self.assertEqual(code,2);call.assert_not_called()

if __name__=='__main__':unittest.main()
