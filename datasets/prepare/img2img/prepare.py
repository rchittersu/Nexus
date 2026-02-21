"""
Image-to-image prepare: (source, target) pairs + captions from JSON -> MDS.

JSON format: array of objects with "source_path", "target_path", "caption".

Example:
  python prepare.py --pairs_json ./pairs.json \\
    --local_mds_dir ./img2img/mds/ --num_proc 4 --seed 42 \\
    --min_size 512
"""

import json
import os
from argparse import ArgumentParser
from multiprocessing import Pool, current_process

import numpy as np
from streaming import MDSWriter
from tqdm import tqdm

from nexus.utils.mds_utils import MDS_IMG2IMG_COLUMNS, load_image_rgb, merge_mds_shards


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser(
        description="Prepare image-to-image pairs for precompute.py (--mode img2img)."
    )
    parser.add_argument(
        "--pairs_json",
        type=str,
        required=True,
        help="Path to JSON file with array of {source_path, target_path, caption}",
    )
    parser.add_argument(
        "--local_mds_dir",
        type=str,
        default="",
        help="Directory to store MDS shards.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=4,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=512,
        help="Minimum dimension (width/height). Set to 0 to disable.",
    )
    return parser.parse_args()


def _current_process_index() -> int:
    p = current_process()
    return p._identity[0] - 1


def _write_shard(items: list[tuple[str, str, str]], args: ArgumentParser) -> None:
    idx = _current_process_index()
    save_dir = os.path.join(args.local_mds_dir, str(idx))
    os.makedirs(save_dir, exist_ok=True)

    writer = MDSWriter(
        out=save_dir,
        columns=MDS_IMG2IMG_COLUMNS,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    for source_path, target_path, caption in tqdm(items, desc=f"Worker {idx}", position=idx):
        try:
            source_img = load_image_rgb(source_path)
            target_img = load_image_rgb(target_path)

            w, h = target_img.size
            if args.min_size > 0 and min(w, h) < args.min_size:
                continue

            cap = (caption or "").strip()
            writer.write(
                {
                    "source_image": source_img,
                    "image": target_img,
                    "caption": cap,
                    "width": w,
                    "height": h,
                }
            )
        except Exception as e:
            print(f"Skipping {source_path} -> {target_path}: {e}")

    writer.finish()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.local_mds_dir, exist_ok=True)

    with open(args.pairs_json) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("pairs_json must contain a JSON array")

    items = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        sp = entry.get("source_path") or entry.get("source")
        tp = entry.get("target_path") or entry.get("target")
        cap = entry.get("caption") or entry.get("prompt") or ""
        if sp and tp:
            items.append((str(sp), str(tp), str(cap)))

    print(f"Loaded {len(items)} pairs from {args.pairs_json}")

    if not items:
        raise ValueError("No valid pairs found in JSON")

    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(items))
    rng.shuffle(indices)
    items = [items[i] for i in indices]

    chunks = np.array_split(np.asarray(items, dtype=object), args.num_proc)
    chunks = [list(c) for c in chunks if len(c) > 0]

    n_workers = len(chunks)
    with Pool(processes=n_workers) as pool:
        pool.starmap(_write_shard, [(c, args) for c in chunks])

    merge_mds_shards(args.local_mds_dir, n_workers)
    print(f"Merged MDS shards to {args.local_mds_dir}")


if __name__ == "__main__":
    main()
