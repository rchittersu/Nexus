"""
DreamBooth prepare: instance (+ class) images -> MDS for precompute.py.

Supports HuggingFace dataset (e.g. diffusers/dog-example) or local folders.

Example (dog from HuggingFace):
  python prepare.py --dataset_name diffusers/dog-example --download_dir ./dog \\
    --instance_prompt "a photo of sks dog" --local_mds_dir ./dreambooth/mds/

Example (local folders):
  python prepare.py --instance_data_dir ./dog --instance_prompt "a photo of sks dog" \\
    --class_data_dir ./dog_class --class_prompt "a dog" --local_mds_dir ./dreambooth/mds/
"""

import os
import sys

# Disable xet storage for HF downloads (must be set before importing huggingface_hub)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Add datasets/ to path for utils import when run as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets.utils import (
    write_path_caption_shard,
    merge_mds_shards,
    collect_images_from_dir,
)


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Prepare DreamBooth instance (and optional class) data for precompute.py."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="HuggingFace dataset (e.g. diffusers/dog-example). Requires --download_dir.",
    )
    parser.add_argument(
        "--download_dir",
        type=str,
        default=None,
        help="Directory to download --dataset_name into. Required when using --dataset_name.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="Local folder with instance images. Mutually exclusive with --dataset_name.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default="a photo of sks dog",
        help="Instance prompt for DreamBooth.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="Local folder with class images for prior preservation. Optional.",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default="a dog",
        help="Class prompt for prior preservation.",
    )
    parser.add_argument(
        "--local_mds_dir",
        type=str,
        required=True,
        help="Directory to store MDS shards (compatible with precompute.py).",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=256,
        help="Minimum dimension. Set to 0 to disable.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat each instance image this many times.",
    )
    return parser.parse_args()


def _download_dataset(dataset_name: str, download_dir: str) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub required. pip install huggingface_hub")
    snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_dir=download_dir,
        ignore_patterns=[".gitattributes"],
    )


def _collect_instance_items(args) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []

    if args.dataset_name is not None:
        if args.download_dir is None:
            raise ValueError("--download_dir is required when using --dataset_name")
        _download_dataset(args.dataset_name, args.download_dir)
        root = Path(args.download_dir)
        for path, w, h in collect_images_from_dir(root, min_size=args.min_size):
            items.append((path, args.instance_prompt))

    elif args.instance_data_dir is not None:
        root = Path(args.instance_data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Instance data dir does not exist: {root}")
        for path, w, h in collect_images_from_dir(root, min_size=args.min_size):
            items.append((path, args.instance_prompt))
    else:
        raise ValueError("Specify either --dataset_name or --instance_data_dir")

    expanded: list[tuple[str, str]] = []
    for path, cap in items:
        for _ in range(args.repeats):
            expanded.append((path, cap))
    return expanded


def _collect_class_items(args) -> list[tuple[str, str]]:
    if args.class_data_dir is None:
        return []
    root = Path(args.class_data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Class data dir does not exist: {root}")
    return [
        (path, args.class_prompt)
        for path, w, h in collect_images_from_dir(root, min_size=args.min_size)
    ]


def main() -> None:
    args = parse_arguments()

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of --dataset_name or --instance_data_dir")

    instance_items = _collect_instance_items(args)
    class_items = _collect_class_items(args)
    all_items = instance_items + class_items
    if not all_items:
        raise ValueError("No instance (or class) images found.")

    n = len(all_items)
    arr = np.arange(n)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(arr)

    partitions = np.array_split(arr, args.num_proc)
    partitions = [p for p in partitions if len(p) > 0]
    num_proc = len(partitions)

    def chunk(i: int) -> list[tuple[str, str]]:
        return [all_items[idx] for idx in partitions[i]]

    if num_proc == 1:
        write_path_caption_shard(chunk(0), args.local_mds_dir, 0)
    else:
        with Pool(processes=num_proc) as pool:
            pool.starmap(
                write_path_caption_shard,
                [(chunk(i), args.local_mds_dir, i) for i in range(num_proc)],
            )

    merge_mds_shards(args.local_mds_dir, num_proc)
    print(f"Wrote {n} samples ({len(instance_items)} instance, {len(class_items)} class) to {args.local_mds_dir}")


if __name__ == "__main__":
    main()
