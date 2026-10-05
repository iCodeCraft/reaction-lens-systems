"""Full-catalog development ranking from a checksummed frozen-feature pack."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from safetensors.torch import load_file

from .artifacts import load_bundle, sha256, verify_files
from .model import LocalCatalogScorer


def ranking_metrics(scores, gold_ids, catalog_ids):
    """AP retains missing-catalog labels in its denominator; ties use catalog order."""
    gold = set(gold_ids)
    if not gold or len(scores) != len(catalog_ids) or not np.isfinite(scores).all():
        raise ValueError('Invalid ranking inputs')
    order = np.argsort(-scores, kind='stable')
    rel = np.array([catalog_ids[i] in gold for i in order], dtype=np.float64)
    ap = float((rel * np.cumsum(rel) / np.arange(1, len(rel) + 1)).sum() / len(gold))
    return ap, int(rel[0]), int(rel[:100].sum()), len(gold)


@torch.inference_mode()
def evaluate_bundle(bundle: Path, evaluation: Path, batch_size: int = 16):
    """CPU check of fresh predictions, never a read-back of saved logits."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    started = time.perf_counter()
    spec, options, catalog = load_bundle(bundle)
    pack = json.loads((evaluation / 'manifest.json').read_text())
    if pack.get('schema_version') != 1 or pack.get('split') != 'development':
        raise ValueError('Unsupported evaluation pack')
    if pack['bundle_sha256'] != sha256(bundle / 'manifest.json'):
        raise ValueError('Evaluation pack belongs to another bundle')
    required = {'articles.npy', 'sentences.npy', 'offsets.npy', 'labels.jsonl'}
    if not required.issubset(pack['files_sha256']):
        raise ValueError('Incomplete evaluation pack')
    verify_files(evaluation, pack['files_sha256'])
    articles, sentences, offsets = [np.load(evaluation/n, allow_pickle=False)
                                   for n in ('articles.npy', 'sentences.npy', 'offsets.npy')]
    labels = [json.loads(x) for x in (evaluation/'labels.jsonl').read_text().splitlines() if x.strip()]
    n, d = len(labels), spec['input_dim']
    if (n != pack['publications'] or n == 0 or articles.shape != (n,d)
            or sentences.ndim != 2 or sentences.shape[1] != d
            or articles.dtype != np.float32 or sentences.dtype != np.float32
            or offsets.shape != (n+1,) or offsets.dtype != np.int64
            or offsets[0] != 0 or offsets[-1] != len(sentences)
            or np.any(np.diff(offsets) <= 0)
            or not np.isfinite(articles).all() or not np.isfinite(sentences).all()
            or len({x['pmid'] for x in labels}) != n):
        raise ValueError('Invalid evaluation features or labels')
    model = LocalCatalogScorer(d, spec['projection_dim'], spec['option_block']).eval()
    model.load_state_dict(load_file(str(bundle/'head.safetensors')), strict=True)
    bank = model.reaction_vectors(torch.from_numpy(catalog))
    ids = [x['reaction_id'] for x in options]
    ap_total = hits = found = gold_count = 0
    for start in range(0,n,batch_size):
        end = min(start+batch_size,n)
        lengths = np.diff(offsets[start:end+1])
        local = np.zeros((end-start,int(lengths.max()),d), dtype=np.float32)
        mask = np.zeros(local.shape[:2], dtype=bool)
        for j,i in enumerate(range(start,end)):
            local[j,:lengths[j]] = sentences[offsets[i]:offsets[i+1]]
            mask[j,:lengths[j]] = True
        logits = model.score_prepared(torch.from_numpy(articles[start:end]),bank,
                                      torch.from_numpy(local),torch.from_numpy(mask)).numpy()
        for score,row in zip(logits,labels[start:end]):
            ap,h,f,g = ranking_metrics(score,row['all_known_reaction_ids'],ids)
            ap_total += ap; hits += h; found += f; gold_count += g
    measured = {'MAP':ap_total/n,'Hit@1':hits/n,'Recall@100':found/gold_count}
    # Fixed release acceptance tolerance: do not let a downloaded pack relax it.
    tolerance = 0.00005
    if set(pack['expected']) != set(measured) or not all(np.isfinite(v) for v in pack['expected'].values()):
        raise ValueError('Invalid reference metrics')
    differences = {k:abs(v-pack['expected'][k]) for k,v in measured.items()}
    result = {'status':'passed' if all(v <= tolerance for v in differences.values()) else 'failed',
              'model_id':spec['model_id'],'publications':n,'catalog_reactions':len(ids),
              'known_links':gold_count,'measured':measured,'expected':pack['expected'],
              'absolute_differences':differences,'absolute_tolerance':tolerance,
              'seconds':time.perf_counter()-started,'scope':pack['scope']}
    return result
