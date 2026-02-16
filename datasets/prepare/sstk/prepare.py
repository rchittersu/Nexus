import json
import os
from argparse import ArgumentParser
from multiprocessing import Pool, current_process

import numpy as np
from PIL import Image
from tqdm import tqdm
from streaming import MDSWriter
from streaming.base.util import merge_index


"""Example usage:
python prepare.py --images_txt ./sa1b/image_paths.txt \
    --local_mds_dir ./sa1b/mds/ --num_proc 16 --seed 42 --size 100000 \
    --min_size 512 --min_aspect_ratio 0.67 --max_aspect_ratio 1.33

Caption path: <image_path>.json, key "description"
"""


def parse_arguments() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument(
        '--images_txt',
        type=str,
        required=True,
        help='Path to txt file with image paths (one path per line)'
    )
    parser.add_argument(
        '--local_mds_dir',
        type=str,
        default='',
        help='Directory to store mds shards.'
    )
    parser.add_argument(
        '--num_proc',
        type=int,
        default=16
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling'
    )
    parser.add_argument(
        '--size',
        type=int,
        default=None,
        help='Number of images to sample. If None, use all images from txt.'
    )
    parser.add_argument(
        '--min_size',
        type=int,
        default=512,
        help='Minimum dimension (width/height). Skip images smaller than this. Set to 0 to disable.',
    )
    parser.add_argument(
        '--min_aspect_ratio',
        type=float,
        default=0.67,
        help='Minimum aspect ratio (width/height). Skip images with ratio below this.',
    )
    parser.add_argument(
        '--max_aspect_ratio',
        type=float,
        default=1.33,
        help='Maximum aspect ratio (width/height). Skip images with ratio above this.',
    )
    args = parser.parse_args()
    return args


def current_process_index() -> int:
    # by default it starts from 1
    p = current_process()
    return p._identity[0] - 1


def write_images(images_path: np.ndarray, args: ArgumentParser) -> None:
    print(f"Writing {len(images_path)} images in the {current_process_index()} proccess")
    assert isinstance(images_path, np.ndarray)
    
    columns = {
        'width': 'int32',
        'height': 'int32',
        'image': 'jpeg',
        'caption': 'str'
    }
    
    save_dir = os.path.join(args.local_mds_dir, str(current_process_index()))
    os.makedirs(save_dir, exist_ok=True)
    
    writer = MDSWriter(
        out=save_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),  # 256MB
        max_workers=64
    )
    
    for f in tqdm(images_path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                img = Image.open(f)
                w, h = img.size
                if args.min_size > 0 and min(w, h) < args.min_size:
                    continue
                aspect_ratio = w / h
                if aspect_ratio < args.min_aspect_ratio or aspect_ratio > args.max_aspect_ratio:
                    continue
                cap_path = str(f) + '.json'
                with open(cap_path, 'r') as cf:
                    cap = json.load(cf)['description']
                cap = str(cap).strip() if cap else ''
                
                mds_sample = {
                    'image': img,
                    'caption': cap,
                    'width': w,
                    'height': h
                }
                writer.write(mds_sample)
                
            except Exception as e:
                print(
                    "Something went wrong in reading image and caption, "
                    f"skipping writing this sample. Error: {e}"
                )
    
    writer.finish()


def main() -> None:
    args = parse_arguments()
    os.makedirs(args.local_mds_dir, exist_ok=True)

    # Read image paths from txt file
    with open(args.images_txt, 'r') as f:
        images_path = [line.strip() for line in f if line.strip()]
    print(f"Total {len(images_path)} images in txt file")

    # Sample based on seed and size
    rng = np.random.default_rng(args.seed)
    if args.size is not None:
        n = min(args.size, len(images_path))
        images_path = rng.choice(images_path, size=n, replace=False).tolist()
        print(f"Sampled {n} images (seed={args.seed}, size={args.size})")
    else:
        rng.shuffle(images_path)
        print(f"Using all {len(images_path)} images (shuffled with seed={args.seed})")

    # use one worker per list of images
    images_path = np.array_split(images_path, args.num_proc)
    
    with Pool(processes=args.num_proc) as pool:
        pool.starmap(
            write_images,
            [(im, args) for im in images_path]
        )
    
    shards_metadata = [
        os.path.join(args.local_mds_dir, str(i), 'index.json')
        for i in range(args.num_proc)
    ]
    merge_index(shards_metadata, out=args.local_mds_dir, keep_local=True)


if __name__ == '__main__':
    main()