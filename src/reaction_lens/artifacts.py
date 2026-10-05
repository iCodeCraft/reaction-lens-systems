"""Portable, checksummed inference bundles. No downloads during inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .decisions import validate_options


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Artifact path escapes its bundle')
    return path


def verify_files(root: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = checked_path(root, name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f'Artifact missing or checksum mismatch: {name}')


def load_bundle(root: Path):
    root = root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('schema_version') != 1 or manifest.get('architecture') != 'local-global-v1':
        raise ValueError('Unsupported inference bundle')
    required = {'head.safetensors', 'catalog.jsonl', 'catalog-vectors.npy'}
    if not required.issubset(manifest['files_sha256']):
        raise ValueError('Incomplete bundle manifest')
    verify_files(root, manifest['files_sha256'])
    options = [json.loads(line) for line in (root / 'catalog.jsonl').read_text().splitlines() if line.strip()]
    validate_options(options)
    vectors = np.load(root / 'catalog-vectors.npy', allow_pickle=False)
    if (vectors.shape != (len(options), manifest['input_dim']) or
            vectors.dtype != np.float32 or not np.isfinite(vectors).all()):
        raise ValueError('Invalid catalog feature matrix')
    if len(options) != manifest['catalog_size'] or not np.isfinite(manifest['logit_threshold']):
        raise ValueError('Invalid catalog size or threshold')
    return manifest, options, vectors


def download_encoder(bundle: Path, destination: Path) -> None:
    """Fetch exact upstream snapshots, then verify trained encoder identities."""
    from huggingface_hub import snapshot_download

    manifest, _, _ = load_bundle(bundle)
    for name in ('base', 'adapter'):
        spec = manifest['encoder'][name]
        if len(spec['revision']) != 40:
            raise ValueError('Encoder revision must be a full commit hash')
        folder = destination / name
        snapshot_download(spec['repo'], revision=spec['revision'], local_dir=folder,
                          allow_patterns=list(spec['files_sha256']))
        verify_files(folder, spec['files_sha256'])


def download_bundle(repo: str, revision: str, destination: Path) -> None:
    """Install a published bundle by immutable Hub revision; no implied default repo."""
    from huggingface_hub import snapshot_download

    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('Use an immutable 40-character Hugging Face commit hash')
    snapshot_download(repo, revision=revision, local_dir=destination,
                      allow_patterns=['manifest.json', 'head.safetensors', 'catalog.jsonl', 'catalog-vectors.npy'])
    load_bundle(destination)
