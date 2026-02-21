"""
Streaming dataset for precompute pipeline: MDS with image(s) and caption.

Supports t2i (image + caption) and img2img (source_image + image + caption).
Used by datasets/precompute.py to stream and transform images before VAE/text encoding.
"""

from collections.abc import Callable, Sequence

from streaming import Stream, StreamingDataset

from .utils import text_preprocessing


class StreamingPrecomputeDataset(StreamingDataset):
    """
    Streaming dataset that resizes images to resolution and tokenizes captions.

    For t2i: yields image, caption. For img2img: also yields source_image when
    source_image_key is set and present in MDS sample.
    """

    def __init__(
        self,
        streams: Sequence[Stream],
        transforms_list: list[Callable],
        batch_size: int,
        shuffle: bool = False,
        image_key: str = "image",
        caption_key: str = "caption",
        clean_caption: bool = False,
        source_image_key: str | None = None,
    ) -> None:
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        assert transforms_list is not None, (
            "Must provide transforms to resize and center crop images"
        )

        self.transforms_list = transforms_list
        self.caption_key = caption_key
        self.image_key = image_key
        self.clean_caption = clean_caption
        self.source_image_key = source_image_key

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)

        ret = {}
        for i, transform in enumerate(self.transforms_list):
            rgb = sample[self.image_key].convert("RGB")
            ret[f"image_{i}"] = transform(rgb)

        if self.source_image_key is not None and self.source_image_key in sample:
            for i, transform in enumerate(self.transforms_list):
                rgb = sample[self.source_image_key].convert("RGB")
                ret[f"source_image_{i}"] = transform(rgb)

        caption = sample[self.caption_key]
        ret["caption"] = text_preprocessing(caption, self.clean_caption)
        ret["sample"] = sample
        return ret
