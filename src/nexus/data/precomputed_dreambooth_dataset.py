"""
Precomputed DreamBooth dataset: instance + class MDS streams for prior preservation.

When with_prior_preservation=True, each __getitem__ returns one instance and one
class sample (paired). Collate stacks pairs: first half of batch = instance,
second half = class. Loss splits accordingly.
"""

import torch
from streaming import StreamingDataset

from .precomputed_sstk_dataset import _bytes_to_latent, _bytes_to_text_embeds


def _decode_sample(
    sample: dict,
    latent_key: str,
    resolution: int,
    latent_channels: int,
    text_embed_hidden: int,
    latent_dtype: torch.dtype,
) -> dict:
    """Decode one MDS sample to latents, text_embeds, text_ids."""
    from diffusers import Flux2KleinPipeline

    latent_bytes = sample[latent_key]
    latents = _bytes_to_latent(
        latent_bytes,
        resolution,
        latent_channels,
        latent_dtype,
    )

    text_embeds = _bytes_to_text_embeds(
        sample["text_embeds"],
        text_embed_hidden,
        latent_dtype,
    )

    # text_ids from text_embeds, same as Flux2KleinPipeline.encode_prompt
    text_embeds_batched = text_embeds.unsqueeze(0)
    text_ids = Flux2KleinPipeline._prepare_text_ids(text_embeds_batched)
    text_ids = text_ids.squeeze(0)

    return {
        "latents": latents,
        "text_embeds": text_embeds,
        "text_ids": text_ids,
        "caption": sample.get("caption", ""),
    }


class PrecomputedDreamBoothDataset(torch.utils.data.Dataset):
    """
    Dataset for DreamBooth training from precomputed instance and class MDS.

    When with_prior_preservation=True, each __getitem__ returns both instance
    and class sample (paired). First half of collated batch = instance,
    second half = class.
    """

    def __init__(
        self,
        local: str,
        instance_subdir: str = "instance",
        class_subdir: str = "class",
        with_prior_preservation: bool = True,
        resolution: int = 512,
        latent_channels: int = 32,
        text_embed_hidden: int = 0,
        batch_size: int = 4,
        shuffle: bool = True,
        latent_dtype: torch.dtype = torch.float32,
    ):
        self.with_prior_preservation = with_prior_preservation
        self.resolution = resolution
        self.latent_channels = latent_channels
        self.text_embed_hidden = text_embed_hidden
        self.latent_dtype = latent_dtype
        self.latent_key = f"latents_{resolution}"

        instance_local = f"{local.rstrip('/')}/{instance_subdir}"
        self.instance_ds = StreamingDataset(
            local=instance_local,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        if with_prior_preservation:
            class_local = f"{local.rstrip('/')}/{class_subdir}"
            self.class_ds = StreamingDataset(
                local=class_local,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            self._len = len(self.instance_ds)
        else:
            self.class_ds = None
            self._len = len(self.instance_ds)

    def __len__(self) -> int:
        return self._len

    def _decode_one(self, sample: dict) -> dict:
        return _decode_sample(
            sample,
            self.latent_key,
            self.resolution,
            self.latent_channels,
            self.text_embed_hidden,
            self.latent_dtype,
        )

    def __getitem__(self, index: int) -> dict:
        inst_idx = index % len(self.instance_ds)
        inst_sample = self.instance_ds[inst_idx]
        inst_decoded = self._decode_one(inst_sample)

        if not self.with_prior_preservation:
            return {
                "latents": inst_decoded["latents"].unsqueeze(0),
                "text_embeds": inst_decoded["text_embeds"].unsqueeze(0),
                "text_ids": inst_decoded["text_ids"].unsqueeze(0),
                "caption": inst_decoded["caption"],
            }

        class_idx = index % len(self.class_ds)
        class_sample = self.class_ds[class_idx]
        class_decoded = self._decode_one(class_sample)

        return {
            "latents": torch.stack([inst_decoded["latents"], class_decoded["latents"]]),
            "text_embeds": torch.stack([inst_decoded["text_embeds"], class_decoded["text_embeds"]]),
            "text_ids": torch.stack([inst_decoded["text_ids"], class_decoded["text_ids"]]),
            "caption": inst_decoded["caption"],
        }


def collate_precomputed_dreambooth(batch: list[dict]) -> dict:
    """Collate DreamBooth batch. Dataset always yields (1,C,H,W) or (2,C,H,W); just concat."""
    if not batch:
        return {}
    n_per = batch[0]["latents"].shape[0]
    return {
        "latents": torch.cat([b["latents"] for b in batch], dim=0),
        "text_embeds": torch.cat([b["text_embeds"] for b in batch], dim=0),
        "text_ids": torch.cat([b["text_ids"] for b in batch], dim=0),
        "captions": [b["caption"] for b in batch for _ in range(n_per)],
    }
