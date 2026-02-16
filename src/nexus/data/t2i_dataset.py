from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from streaming import Stream, StreamingDataset


from .utils import text_preprocessing

class StreamingT2IDataset(StreamingDataset):
    """Streaming dataset that resizes images to user-provided resolutions and tokenizes captions."""

    def __init__(
        self,
        streams: Sequence[Stream],
        transforms_list: List[Callable],
        batch_size: int,
        shuffle: bool = False,
        image_key: str = 'image',
        caption_key: str = 'caption',
        clean_caption: bool = False,
    ) -> None:
        super().__init__(
            streams=streams,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        assert transforms_list is not None, 'Must provide transforms to resize and center crop images'

        self.transforms_list = transforms_list
        self.caption_key = caption_key
        self.image_key = image_key
        self.clean_caption = clean_caption

    def __getitem__(self, index: int) -> Dict:
        sample = super().__getitem__(index)

        # process images    
        ret = {f'image_{i}': transform(sample[self.image_key].convert('RGB')) for i, transform in enumerate(self.transforms_list)}

        # process caption
        caption = sample[self.caption_key]
        ret["caption"] = text_preprocessing(caption, self.clean_caption)
        ret['sample'] = sample
        return ret