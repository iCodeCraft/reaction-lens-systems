"""Measure the released inference system without training or accessing test data.

CPU only. Cached-feature timings exclude encoding, loading and serialization.
Optional end-to-end measurements include encoding and the complete result payload.
All repetitions and numerical contract checks are retained in JSON.
"""
import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import torch
from safetensors.torch import load_file
from reaction_lens.artifacts import load_bundle, sha256, verify_files
from reaction_lens.model import LocalCatalogScorer


def rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def describe(values):
    a = np.asarray(values, dtype=float)
    return {'count': len(a), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95)), 'min': float(a.min()), 'max': float(a.max())}


def environment():
    chip = platform.processor()
    if sys.platform == 'darwin':
        try:
            chip = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            chip = platform.machine() + ' (CPU brand unavailable in sandbox)'
    return {'os': platform.platform(), 'cpu': chip, 'python': platform.python_version(),
            'torch': torch.__version__, 'numpy': np.__version__,
            'device': 'cpu', 'threads': torch.get_num_threads(), 'dtype': 'float32'}


@torch.inference_mode()
def worker(args):
    spec, options, catalog = load_bundle(args.bundle)
    folder = args.bundle / 'evaluation'
    pack = json.loads((folder / 'manifest.json').read_text())
    if pack['split'] != 'development' or pack['bundle_sha256'] != sha256(args.bundle / 'manifest.json'):
        raise ValueError('Expected development features for this exact bundle')
    verify_files(folder, pack['files_sha256'])
    articles, sentences, offsets = [np.load(folder / name, allow_pickle=False)
        for name in ('articles.npy', 'sentences.npy', 'offsets.npy')]
    labels = [json.loads(line) for line in (folder / 'labels.jsonl').read_text().splitlines()]
    rng = np.random.default_rng(20261005)
    indices = sorted(rng.choice(len(articles), min(args.samples, len(articles)), replace=False).tolist())
    model = LocalCatalogScorer(spec['input_dim'], spec['projection_dim'], args.worker).eval()
    model.load_state_dict(load_file(str(args.bundle / 'head.safetensors')), strict=True)
    model.requires_grad_(False)
    bank = model.reaction_vectors(torch.from_numpy(catalog))

    def query(i):
        local = torch.from_numpy(sentences[offsets[i]:offsets[i + 1]])[None]
        return torch.from_numpy(articles[i:i+1]), local, torch.ones(local.shape[:2], dtype=torch.bool)

    if args.memory_only:
        chosen = indices[:16]
        lengths = [int(offsets[i+1]-offsets[i]) for i in chosen]
        local = torch.zeros(len(chosen), max(lengths), spec['input_dim'])
        mask = torch.zeros(local.shape[:2], dtype=torch.bool)
        for j, i in enumerate(chosen):
            local[j, :lengths[j]] = torch.from_numpy(sentences[offsets[i]:offsets[i+1]])
            mask[j, :lengths[j]] = True
        before = rss_bytes()
        out = model.score_prepared(torch.from_numpy(articles[chosen]), bank, local, mask)
        return {'block_size': args.worker, 'environment': environment(),
                'process_peak_rss_bytes': rss_bytes(), 'pre_scoring_peak_rss_bytes': before,
                'batch': len(chosen), 'max_sentences': max(lengths),
                'output_shape': list(out.shape), 'all_finite': bool(out.isfinite().all())}
    for _ in range(5):
        a, s, m = query(indices[0]); model.score_prepared(a, bank, s, m)
    timings, contracts = [], []
    for repeat in range(args.repeats):
        for i in indices:
            a, s, m = query(i)
            start = time.perf_counter()
            score = model.score_prepared(a, bank, s, m)[0]
            elapsed = time.perf_counter() - start
            timings.append({'repeat': repeat, 'row': i, 'pmid': labels[i]['pmid'],
                            'sentences': s.shape[1], 'seconds': elapsed})
            if repeat:
                continue
            permutation = torch.from_numpy(rng.permutation(len(bank)))
            subset = permutation[:len(bank)//2]
            # All scenarios reuse the same query, weights and reaction vectors.
            cases = [('permutation', bank[permutation], score[permutation]),
                     ('subset', bank[subset], score[subset]),
                     ('append_duplicates', torch.cat([bank, bank[:128]]), score)]
            for name, candidates, expected in cases:
                actual = model.score_prepared(a, candidates, s, m)[0][:len(expected)]
                contracts.append({'row': i, 'scenario': name,
                    'max_absolute_logit_error': float((actual - expected).abs().max()),
                    'selection_flips': int(((actual >= spec['logit_threshold']) !=
                                           (expected >= spec['logit_threshold'])).sum()),
                    'comparisons': len(expected)})
            original_block = model.option_block
            model.option_block = len(bank)
            reference = model.score_prepared(a, bank, s, m)[0]
            model.option_block = original_block
            contracts.append({'row': i, 'scenario': 'unblocked_reference',
                'max_absolute_logit_error': float((score-reference).abs().max()),
                'selection_flips': int(((score >= spec['logit_threshold']) !=
                                       (reference >= spec['logit_threshold'])).sum()),
                'comparisons': len(bank)})
    # Memory measurement includes a padded batch, in a separate process per block size.
    chosen = indices[:16]
    lengths = [int(offsets[i+1]-offsets[i]) for i in chosen]
    local = torch.zeros(len(chosen), max(lengths), spec['input_dim'])
    mask = torch.zeros(local.shape[:2], dtype=torch.bool)
    for j, i in enumerate(chosen):
        local[j, :lengths[j]] = torch.from_numpy(sentences[offsets[i]:offsets[i+1]])
        mask[j, :lengths[j]] = True
    model.score_prepared(torch.from_numpy(articles[chosen]), bank, local, mask)
    return {'block_size': args.worker, 'environment': environment(),
        'sample_indices': indices, 'timings': timings, 'contracts': contracts,
        'latency_seconds': describe([x['seconds'] for x in timings]),
        'process_peak_rss_bytes': rss_bytes(), 'memory_batch': len(chosen),
        'memory_max_sentences': max(lengths), 'catalog_size': len(options)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=64)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--worker', type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument('--memory-only', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--encoder', type=Path)
    p.add_argument('--examples', type=Path)
    args = p.parse_args()
    if min(args.samples, args.repeats, args.threads) < 1 or bool(args.encoder) != bool(args.examples):
        p.error('Positive counts required; use --encoder and --examples together')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    if args.worker:
        result = worker(args)
    else:
        spec, _, _ = load_bundle(args.bundle)
        result = {'schema_version': 1, 'scope': 'Single-host engineering measurements on development inputs; no training, independent test, biological validation or new calibration.',
            'protocol': {'seed': 20261005, 'samples': args.samples, 'repeats': args.repeats,
                         'warmup_queries': 5, 'absolute_logit_tolerance': 1e-5,
                         'fixed_threshold': spec['logit_threshold'],
                         'memory_scope': 'memory_runs: fresh subprocess per block size, imports/artifacts and one padded batch; no unblocked reference or contract probes. Whole-process high-water RSS, not incremental tensor allocation. cached_feature_runs RSS includes probes and must not be used for memory comparison.'},
            'bundle_sha256': sha256(args.bundle/'manifest.json'),
            'evaluation_sha256': sha256(args.bundle/'evaluation/manifest.json'),
            'source_sha256': {name: sha256(ROOT/name) for name in
                ['scripts/benchmark_system.py', 'src/reaction_lens/model.py', 'src/reaction_lens/pipeline.py', 'src/reaction_lens/encoder.py']},
            'cached_feature_runs': [], 'memory_runs': []}
        for block in (128, 1024, spec['catalog_size']):
            temp = args.output.with_suffix(f'.block{block}.json')
            subprocess.run([sys.executable, __file__, '--bundle', str(args.bundle),
                '--output', str(temp), '--samples', str(args.samples), '--repeats', str(args.repeats),
                '--threads', str(args.threads), '--worker', str(block)], check=True)
            result['cached_feature_runs'].append(json.loads(temp.read_text()))
            temp.unlink()
            subprocess.run([sys.executable, __file__, '--bundle', str(args.bundle),
                '--output', str(temp), '--samples', str(args.samples), '--threads', str(args.threads),
                '--worker', str(block), '--memory-only'], check=True)
            result['memory_runs'].append(json.loads(temp.read_text()))
            temp.unlink()
        if args.encoder:
            from reaction_lens.pipeline import ReactionLens
            examples = json.loads(args.examples.read_text())['examples']
            started = time.perf_counter()
            model = ReactionLens.from_bundle(args.bundle, args.encoder, 'cpu')
            setup = time.perf_counter() - started
            model.predict(examples[0]['title'], examples[0]['abstract'])
            rows = []
            for repeat in range(args.repeats):
                for row in examples:
                    started = time.perf_counter()
                    output = model.predict(row['title'], row['abstract'])
                    predict_seconds = time.perf_counter()-started
                    encoded = json.dumps(output).encode()
                    rows.append({'repeat': repeat, 'pmid': row['pmid'],
                        'sentence_count': len(output['sentence_spans']),
                        'options_scored': output['options_scored'],
                        'predict_seconds': predict_seconds,
                        'predict_plus_json_seconds': time.perf_counter()-started,
                        'json_bytes': len(encoded), **output['timings']})
            result['end_to_end'] = {'environment': environment(), 'setup_seconds': setup,
                'examples_sha256': sha256(args.examples), 'warmup_queries': 1,
                'selection': 'All existing released comparison examples; convenience sample, not representative latency or semantic generalization benchmark.',
                'rows': rows, 'predict_seconds': describe([r['predict_seconds'] for r in rows]),
                'process_peak_rss_bytes': rss_bytes()}
        checks = [c for r in result['cached_feature_runs'] for c in r['contracts']]
        result['contract_summary'] = {'checks': len(checks),
            'score_comparisons': sum(c['comparisons'] for c in checks),
            'max_absolute_logit_error': max(c['max_absolute_logit_error'] for c in checks),
            'selection_flips': sum(c['selection_flips'] for c in checks)}
        result['status'] = 'passed' if (result['contract_summary']['max_absolute_logit_error'] <= 1e-5
            and result['contract_summary']['selection_flips'] == 0) else 'failed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    if result.get('status') == 'failed':
        raise SystemExit('Contract tolerance exceeded; preserve report and investigate')


if __name__ == '__main__':
    main()
