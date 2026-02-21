"""
Precomputed SSTK dataset: MDS shards with VAE latents and text embeddings.

Produced by datasets/precompute.py. Schema: caption, latents_{resolution},
text_embeds. Single resolution (default 512). text_ids generated from text_embeds
via Flux2KleinPipeline._prepare_text_ids. Enables training without running VAE
or text encoder during training.

For img2img MDS (from precompute --mode img2img): when source_latents_{resolution}
exists, optionally returns source_latents (raw, same format as target).
"""

import numpy as np
import torch
from streaming import StreamingDataset


def _bytes_to_latent(
    data: bytes,
    resolution: int,
    latent_channels: int = 32,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Decode latent bytes to tensor. FLUX.2 VAE: 8x downscale, 32 channels."""
    h = w = resolution // 8  # FLUX VAE 8x downscale
    shape = (latent_channels, h, w)
    arr = np.frombuffer(data, dtype=np.float16)
    return torch.from_numpy(arr.reshape(shape).astype(np.float32)).to(dtype)


def _bytes_to_text_embeds(
    data: bytes,
    hidden_dim: int,
    dtype: torch.dtype = torch.float32,
    default_seq_len: int = 512,
) -> torch.Tensor:
    """Decode text_embeds bytes to tensor (seq_len, hidden_dim)."""
    arr = np.frombuffer(data, dtype=np.float16)
    n = len(arr)
    if hidden_dim <= 0:
        hidden_dim = n // default_seq_len
        seq_len = default_seq_len
    else:
        seq_len = n // hidden_dim
    return torch.from_numpy(arr.reshape(seq_len, hidden_dim).astype(np.float32)).to(dtype)


class PrecomputedSSTKDataset(StreamingDataset):
    """
    StreamingDataset for MDS shards with precomputed VAE latents and text embeddings.

    Produced by datasets/precompute.py.
    Uses single resolution via latents_{resolution} (default 512).
    """

    def __init__(
        self,
        local: str,
        resolution: int = 512,
        latent_channels: int = 32,
        text_embed_hidden: int = 0,
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
        self.text_embed_hidden = text_embed_hidden
        self.latent_dtype = latent_dtype
        self.latent_key = f"latents_{resolution}"
        self.source_latent_key = f"source_latents_{resolution}"

    def __getitem__(self, index: int) -> dict:
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
            self.text_embed_hidden,  # 0 = auto-infer from byte size
            self.latent_dtype,
        )

        # text_ids from text_embeds, same as Flux2KleinPipeline.encode_prompt
        from diffusers import Flux2KleinPipeline

        text_embeds_batched = text_embeds.unsqueeze(0)
        text_ids = Flux2KleinPipeline._prepare_text_ids(text_embeds_batched)
        text_ids = text_ids.squeeze(0)

        out = {
            "latents": latents,
            "text_embeds": text_embeds,
            "text_ids": text_ids,
            "caption": sample.get("caption", ""),
        }
        if self.source_latent_key in sample:
            source_latents = _bytes_to_latent(
                sample[self.source_latent_key],
                self.resolution,
                self.latent_channels,
                self.latent_dtype,
            )
            out["source_latents"] = source_latents
        return out


def collate_precomputed(batch: list[dict]) -> dict:
    """Collate precomputed batch for training. Adds source_latents when present (img2img)."""
    out = {
        "latents": torch.stack([b["latents"] for b in batch]),
        "text_embeds": torch.stack([b["text_embeds"] for b in batch]),
        "text_ids": torch.stack([b["text_ids"] for b in batch]),
        "captions": [b["caption"] for b in batch],
    }
    if batch and "source_latents" in batch[0]:
        out["source_latents"] = torch.stack([b["source_latents"] for b in batch])
    return out
