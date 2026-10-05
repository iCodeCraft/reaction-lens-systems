"""Verify paper evidence, optionally recompute model ranking from cached features."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', help='HF model repository; private repos require hf auth login')
    p.add_argument('--revision', help='Full immutable HF commit SHA')
    p.add_argument('--bundle', type=Path, help='Existing local bundle; no downloads')
    p.add_argument('--cache', type=Path, default=ROOT/'artifacts/hub')
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if bool(a.repo) != bool(a.revision) or (a.repo and a.bundle):
        p.error('Use --repo with --revision, or --bundle')
    if a.threads < 1:
        p.error('--threads must be positive')
    if not a.repo and not a.bundle:
        p.error('Supply --bundle or --repo/--revision for fresh model evaluation')
    sys.path.insert(0,str(ROOT/'src'))
    import torch
    from reaction_lens.artifacts import download_bundle
    from reaction_lens.evaluation import evaluate_bundle
    if a.repo:
        from huggingface_hub import snapshot_download
        # Revision validation and bundle checks happen before extra downloads.
        a.bundle = a.cache / a.repo.replace('/','--') / a.revision
        download_bundle(a.repo,a.revision,a.bundle)
        snapshot_download(a.repo,revision=a.revision,local_dir=a.bundle,
                          allow_patterns=['evaluation/manifest.json','evaluation/articles.npy',
                                          'evaluation/sentences.npy','evaluation/offsets.npy','evaluation/labels.jsonl'])
    if not (a.bundle/'manifest.json').is_file():
        p.error('Bundle manifest missing; run make artifacts or supply a complete --bundle')
    torch.set_num_threads(a.threads)
    result = evaluate_bundle(a.bundle,a.bundle/'evaluation')
    print(json.dumps(result,indent=2))
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
