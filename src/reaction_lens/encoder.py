"""Frozen SPECTER2 encoding with the exact research chunking/pooling policy."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .artifacts import verify_files


class SpecterEncoder:
    """510-token chunks → mean CLS → L2 normalization; no discarded text tokens."""

    def __init__(self, spec: dict, root: Path, device: str = 'cpu', batch_size: int = 4):
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        for name in ('base', 'adapter'):
            verify_files(root / name, spec[name]['files_sha256'])
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(root / 'base', local_files_only=True)
        self.model = AutoAdapterModel.from_pretrained(root / 'base', local_files_only=True)
        self.model.load_adapter(str(root / 'adapter'), load_as='proximity', set_active=True)
        self.model.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            raise ValueError('No texts supplied')
        flat, counts = [], []
        for title, abstract in pairs:
            tokens = self.tokenizer.encode(title + self.tokenizer.sep_token + abstract,
                                           add_special_tokens=False)
            chunks = [tokens[i:i + 510] for i in range(0, len(tokens), 510)] or [[]]
            counts.append(len(chunks))
            flat.extend(chunks)
        encoded = []
        for start in range(0, len(flat), self.batch_size):
            inputs = self.tokenizer.pad([
                {'input_ids': self.tokenizer.build_inputs_with_special_tokens(chunk)}
                for chunk in flat[start:start + self.batch_size]
            ], padding=True, return_tensors='pt').to(self.device)
            encoded.append(self.model(**inputs).last_hidden_state[:, 0].float().cpu().numpy())
        encoded = np.concatenate(encoded)
        vectors, start = [], 0
        for count in counts:
            vector = encoded[start:start + count].mean(axis=0)
            start += count
            vectors.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
        return np.stack(vectors).astype(np.float32)
