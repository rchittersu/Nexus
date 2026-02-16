from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from streaming import Stream, StreamingDataset

from transformers import AutoTokenizer

from .utils import text_preprocessing

class StreamingT2IDataset(StreamingDataset):
    """Streaming dataset that resizes images to user-provided resolutions and tokenizes captions."""

    def __init__(
        self,
        streams: Sequence[Stream],
        transforms_list: List[Callable],
        tokenizer: AutoTokenizer,
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

        assert self.transforms_list is not None, 'Must provide transforms to resize and center crop images'
        assert tokenizer is not None, 'Must provide a tokenizer'

        self.transforms_list = transforms_list
        self.caption_key = caption_key
        self.image_key = image_key
        self.tokenizer = tokenizer
        self.clean_caption = clean_caption

    def __getitem__(self, index: int) -> Dict:
        sample = super().__getitem__(index)

        # process images    
        ret = {f'image_{i}': transform(sample[self.image_key].convert('RGB')) for i, transform in enumerate(self.transforms_list)}

        # process caption
        caption = sample[self.caption_key]
        ret["caption"] = caption

        out = self.tokenizer.tokenize(text_preprocessing(caption, self.clean_caption))
        ret["input_ids"] = out['input_ids'].clone().detach()
        if 'attention_mask' in out:
            ret["attention_mask"] = out['attention_mask'].clone().detach()

        ret['sample'] = sample
        return ret