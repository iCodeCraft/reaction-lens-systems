"""Contract checks using a clearly synthetic tiny fixture, never benchmark evidence."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from reaction_lens.artifacts import load_bundle, sha256
from reaction_lens.model import LocalCatalogScorer
from reaction_lens.pipeline import ReactionLens
from reaction_lens.server import create_app


class TinyEncoder:
    def encode(self, pairs):
        return np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (len(pairs), 1))


class ServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        torch.manual_seed(7)
        model = LocalCatalogScorer(4, 2, 2)
        save_file(model.state_dict(), str(self.root / 'head.safetensors'))
        options = [{'reaction_id': f'option-{i}', 'description': '=SUM(1,2)' if i == 0 else f'Option {i}'} for i in range(5)]
        (self.root / 'catalog.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in options))
        np.save(self.root / 'catalog-vectors.npy', np.eye(5, 4, dtype=np.float32))
        manifest = {'schema_version': 1, 'architecture': 'local-global-v1', 'model_id': 'synthetic-test',
                    'input_dim': 4, 'projection_dim': 2, 'option_block': 2, 'catalog_size': 5,
                    'logit_threshold': 0., 'calibration': 'synthetic fixture', 'encoder': {},
                    'files_sha256': {n: sha256(self.root / n) for n in ['head.safetensors', 'catalog.jsonl', 'catalog-vectors.npy']}}
        (self.root / 'manifest.json').write_text(json.dumps(manifest))
        self.pipeline = ReactionLens(self.root, TinyEncoder())

    def tearDown(self):
        self.temp.cleanup()

    def test_full_catalog_and_json_csv_contract(self):
        with TestClient(create_app(self.pipeline)) as client:
            self.assertEqual(client.get('/readyz').status_code, 200)
            self.assertIn('One abstract.', client.get('/').text)
            self.assertEqual(client.get('/static/app.js').status_code, 200)
            response = client.post('/api/predict', json={'abstract': 'One sentence.', 'threshold': -100})
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data['options_scored'], 5)
            self.assertEqual(len(data['proposed_reaction_ids']), 5)
            self.assertEqual(sorted(r['rank'] for r in data['options']), [1, 2, 3, 4, 5])
            exported = client.post('/api/predict?format=csv', json={'title': 'Query'})
            rows = list(csv.DictReader(io.StringIO(exported.text)))
            self.assertEqual(len(rows), 5)
            self.assertEqual(next(r for r in rows if r['reaction_id'] == 'option-0')['description'], "'=SUM(1,2)")
            empty = client.post('/api/predict', json={'title': 'Query', 'threshold': 100}).json()
            self.assertEqual(empty['proposed_reaction_ids'], [])

    def test_invalid_inputs_and_busy_are_explicit(self):
        with TestClient(create_app(self.pipeline)) as client:
            for body in ({}, {'abstract': ' '}, {'abstract': 'x' * 50001}, {'title': 'x', 'unknown': 1}):
                self.assertEqual(client.post('/api/predict', json=body).status_code, 422)
            self.pipeline._lock.acquire()
            try:
                response = client.post('/api/predict', json={'title': 'Query'})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.headers['retry-after'], '2')
            finally:
                self.pipeline._lock.release()

    def test_unconfigured_server_has_no_fake_predictions(self):
        with TestClient(create_app()) as client:
            self.assertEqual(client.get('/healthz').status_code, 200)
            self.assertEqual(client.get('/readyz').status_code, 503)
            self.assertEqual(client.post('/api/predict', json={'title': 'Query'}).status_code, 503)

    def test_modified_artifact_is_rejected(self):
        with (self.root / 'catalog.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_bundle(self.root)


if __name__ == '__main__':
    unittest.main()
