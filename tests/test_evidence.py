"""Reject materially corrupted evidence even after tables have been regenerated."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('verify_study', ROOT/'scripts/verify_study.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class EvidenceTests(unittest.TestCase):
    def test_corruption_is_rejected(self):
        mutations = {
            'duplicate probe': lambda r: r['cached_feature_runs'][0]['contracts'].__setitem__(4, r['cached_feature_runs'][0]['contracts'][0]),
            'wrong denominator': lambda r: r['cached_feature_runs'][0]['contracts'][0].__setitem__('comparisons', 1),
            'negative error': lambda r: r['cached_feature_runs'][0]['contracts'][0].__setitem__('max_absolute_logit_error', -1),
            'changed sample': lambda r: r['cached_feature_runs'][1]['sample_indices'].__setitem__(0, -1),
            'missing block': lambda r: r['memory_runs'].pop(),
            'wrong memory shape': lambda r: r['memory_runs'][0].__setitem__('output_shape', [1, 1]),
            'relaxed tolerance': lambda r: r['protocol'].__setitem__('absolute_logit_tolerance', 1),
            'wrong bundle': lambda r: r.__setitem__('bundle_sha256', '0'*64),
            'partial prediction': lambda r: r['end_to_end']['rows'][0].__setitem__('options_scored', 1),
            'duplicate prediction': lambda r: r['end_to_end']['rows'].__setitem__(1, r['end_to_end']['rows'][0]),
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['measured-source-hashes.json', 'artifact-manifest.json',
                         'results/fresh-evaluation.json', 'examples/latency-examples.json',
                         *json.loads((ROOT/'measured-source-hashes.json').read_text())]:
                dest = root/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, dest)
            verifier.ROOT = root
            try:
                for name, mutate in mutations.items():
                    with self.subTest(name=name):
                        raw = json.loads((ROOT/'results/system-benchmark.json').read_text())
                        mutate(raw)
                        (root/'results/system-benchmark.json').write_text(json.dumps(raw))
                        with self.assertRaises(SystemExit):
                            verifier.main()
            finally:
                verifier.ROOT = ROOT

    def test_missing_metric_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['measured-source-hashes.json', 'artifact-manifest.json',
                         'results/system-benchmark.json', 'examples/latency-examples.json',
                         *json.loads((ROOT/'measured-source-hashes.json').read_text())]:
                dest = root/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, dest)
            ev = json.loads((ROOT/'results/fresh-evaluation.json').read_text())
            ev['measured'] = {}
            (root/'results/fresh-evaluation.json').write_text(json.dumps(ev))
            verifier.ROOT = root
            try:
                with self.assertRaisesRegex(SystemExit, 'Missing metric'):
                    verifier.main()
            finally:
                verifier.ROOT = ROOT


if __name__ == '__main__':
    unittest.main()
