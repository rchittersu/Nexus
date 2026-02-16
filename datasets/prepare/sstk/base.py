from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from streaming import Stream, StreamingDataset

from nexus.data.t2i_dataset import StreamingT2IDataset
from transformers import AutoTokenizer


def build_streaming_sstk_t2i_dataloader(
    datadir: Union[List[str], str],
    batch_size: int,
    resize_sizes: Optional[List[int]] = None,
    drop_last: bool = False,
    shuffle: bool = True,
    image_key: str = 'image',
    caption_key: str = 'caption',
    clean_caption: bool = True,
    tokenizer: AutoTokenizer = None,
    **dataloader_kwargs,
) -> DataLoader:
    assert resize_sizes is not None, 'Must provide target resolution for image resizing'
    assert tokenizer is not None, 'Must provide a tokenizer'

    transforms_list = [
        transforms.Compose([
                transforms.Resize(
                    size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        for size in resize_sizes
    ]

    dataset = StreamingT2IDataset(
        streams=datadir,
        transforms_list=transforms_list,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=shuffle,
        image_key=image_key,
        caption_key=caption_key,
        clean_caption=clean_caption,
    )

    def custom_collate(batch_items: List[Dict]) -> Dict:
        out = {k: [] for k in batch_items[0].keys()}
        for item in batch_items:
            for key, value in item.items():
                out[key].append(value)
        return out

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=custom_collate,
        **dataloader_kwargs,
    )

    return dataloader