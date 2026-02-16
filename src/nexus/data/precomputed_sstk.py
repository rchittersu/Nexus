"""
Precomputed SSTK dataset: MDS shards with VAE latents and text embeddings.

Produced by datasets/prepare/sstk/precompute.py. Schema: caption, latents_512
(or latents_1024, etc.), text_embeds, text_ids. Enables training without
running VAE or text encoder during training.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from streaming import StreamingDataset


def _bytes_to_latent(
    data: bytes,
    resolution: int,
    latent_channels: int = 16,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode latent bytes to tensor. FLUX VAE: 8x downscale, 16 channels."""
    h = w = resolution // 8  # FLUX VAE 8x downscale
    shape = (latent_channels, h, w)
    arr = np.frombuffer(data, dtype=np.float16)
    return torch.from_numpy(arr.reshape(shape).astype(np.float32)).to(dtype)


def _bytes_to_text_embeds(
    data: bytes,
    seq_len: int,
    hidden_dim: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode text_embeds bytes to tensor (seq_len, hidden_dim)."""
    arr = np.frombuffer(data, dtype=np.float16)
    n = len(arr)
    if hidden_dim <= 0:
        # Auto-infer: n = seq_len * hidden_dim
        hidden_dim = n // seq_len
    return torch.from_numpy(arr.reshape(seq_len, hidden_dim).astype(np.float32)).to(dtype)


def _bytes_to_text_ids(
    data: bytes,
    seq_len: int,
    ids_dim: int = 4,
) -> torch.Tensor:
    """Decode text_ids bytes to tensor (seq_len, ids_dim). Flux uses 4D position ids."""
    arr = np.frombuffer(data, dtype=np.int8)
    return torch.from_numpy(arr.reshape(seq_len, ids_dim).astype(np.int64))


class PrecomputedSSTKDataset(StreamingDataset):
    """
    StreamingDataset for MDS shards with precomputed VAE latents and text embeddings.

    Produced by datasets/prepare/sstk/precompute.py.
    Supports multiple resolutions via latents_512, latents_1024, etc.
    """

    def __init__(
        self,
        local: str,
        resolution: int = 512,
        latent_channels: int = 16,
        text_embed_seq_len: int = 512,
        text_embed_hidden: int = 3584,
        text_ids_dim: int = 4,
        batch_size: int = 4,
        shuffle: bool = True,
        latent_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            local=local,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.resolution = resolution
        self.latent_channels = latent_channels
        self.text_embed_seq_len = text_embed_seq_len
        self.text_embed_hidden = text_embed_hidden
        self.text_ids_dim = text_ids_dim
        self.latent_dtype = latent_dtype
        self.latent_key = f"latents_{resolution}"

    def __getitem__(self, index: int) -> Dict:
        sample = super().__getitem__(index)

        latent_bytes = sample[self.latent_key]
        latents = _bytes_to_latent(
            latent_bytes,
            self.resolution,
            self.latent_channels,
            self.latent_dtype,
        )

        text_embeds = _bytes_to_text_embeds(
            sample["text_embeds"],
            self.text_embed_seq_len,
            self.text_embed_hidden,  # 0 = auto-infer from byte size
            self.latent_dtype,
        )

        text_ids = _bytes_to_text_ids(
            sample["text_ids"],
            self.text_embed_seq_len,
            self.text_ids_dim,
        )

        return {
            "latents": latents,
            "text_embeds": text_embeds,
            "text_ids": text_ids,
            "caption": sample.get("caption", ""),
        }


def collate_precomputed(batch: List[Dict]) -> Dict:
    """Collate precomputed batch for training."""
    return {
        "latents": torch.stack([b["latents"] for b in batch]),
        "text_embeds": torch.stack([b["text_embeds"] for b in batch]),
        "text_ids": torch.stack([b["text_ids"] for b in batch]),
        "captions": [b["caption"] for b in batch],
    }
