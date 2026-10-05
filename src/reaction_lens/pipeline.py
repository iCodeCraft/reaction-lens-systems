"""One inference contract for Python, CLI and the local playground."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from safetensors.torch import load_file

from .artifacts import load_bundle, sha256
from .decisions import decision_payload
from .model import LocalCatalogScorer
from .text import segment_texts, segments


class Encoder(Protocol):
    def encode(self, pairs: list[tuple[str, str]]) -> np.ndarray: ...


class BusyError(RuntimeError):
    """A single-process model is currently handling another request."""


class ReactionLens:
    """Read-only model serving a complete, versioned reaction catalog.

    Parameters are loaded once. Selection uses the bundle's development-fitted
    raw-logit threshold; sigmoid scores are not biological probabilities.
    """

    def __init__(self, bundle: str | Path, encoder: Encoder, device: str = 'cpu'):
        self.root = Path(bundle)
        self.manifest, self.options, vectors = load_bundle(self.root)
        self.identity = sha256(self.root / 'manifest.json')
        self.encoder = encoder
        self.device = torch.device(device)
        self.model = LocalCatalogScorer(self.manifest['input_dim'],
                                       self.manifest['projection_dim'],
                                       self.manifest['option_block']).to(self.device).eval()
        self.model.load_state_dict(load_file(str(self.root / 'head.safetensors'), device='cpu'), strict=True)
        self.model.requires_grad_(False)
        with torch.inference_mode():
            self.bank = self.model.reaction_vectors(torch.from_numpy(vectors).to(self.device))
        self._lock = threading.Lock()

    @classmethod
    def from_bundle(cls, bundle: str | Path, encoder_dir: str | Path, device: str = 'cpu'):
        from .encoder import SpecterEncoder

        root = Path(bundle)
        manifest, _, _ = load_bundle(root)
        return cls(root, SpecterEncoder(manifest['encoder'], Path(encoder_dir), device), device)

    def metadata(self) -> dict:
        return {'model_id': self.manifest['model_id'], 'bundle_sha256': self.identity,
                'catalog_size': len(self.options), 'device': str(self.device),
                'architecture': self.manifest['architecture'],
                'logit_threshold': self.manifest['logit_threshold'],
                'calibration': self.manifest['calibration'],
                'encoder': {k: {n: v[n] for n in ('repo', 'revision')}
                            for k, v in self.manifest['encoder'].items()}}

    def predict(self, title: str = '', abstract: str = '', threshold: float | None = None) -> dict:
        if not isinstance(title, str) or not isinstance(abstract, str):
            raise ValueError('Title and abstract must be strings')
        if not (title.strip() or abstract.strip()):
            raise ValueError('Provide a title or abstract')
        if len(title) > 2000 or len(abstract) > 50000:
            raise ValueError('Maximum lengths: title 2000, abstract 50000 characters')
        cutoff = self.manifest['logit_threshold'] if threshold is None else threshold
        if not np.isfinite(cutoff):
            raise ValueError('Threshold must be finite')
        if not self._lock.acquire(blocking=False):
            raise BusyError('Model is busy; retry after the current request')
        try:
            return self._predict(title, abstract, float(cutoff), threshold is not None)
        finally:
            self._lock.release()

    @torch.inference_mode()
    def _predict(self, title: str, abstract: str, cutoff: float, override: bool) -> dict:
        started = time.perf_counter()
        row = {'title': title, 'abstract': abstract}
        spans = segments(row)
        vectors = self.encoder.encode([(title, abstract)] + [(s, '') for s in segment_texts(row, spans)])
        if vectors.shape != (1 + len(spans), self.manifest['input_dim']) or not np.isfinite(vectors).all():
            raise ValueError('Encoder returned invalid features')
        encoded_at = time.perf_counter()
        article = torch.from_numpy(vectors[:1]).to(self.device)
        local = torch.from_numpy(vectors[1:][None]).to(self.device)
        mask = torch.ones(local.shape[:2], dtype=torch.bool, device=self.device)
        logits = self.model.score_prepared(article, self.bank, local, mask)[0].cpu().numpy()
        scored_at = time.perf_counter()
        result = decision_payload(self.options, logits, cutoff)
        result.update(title=title, model=self.metadata(), sentence_spans=spans,
                      selection_policy='user override' if override else 'full-development fitted threshold',
                      timings={'encoding_seconds': encoded_at - started,
                               'scoring_seconds': scored_at - encoded_at,
                               'request_seconds': time.perf_counter() - started},
                      interpretation='Proposals for review. Citation labels are incomplete; scores are not biological probabilities.')
        return result
