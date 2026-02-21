"""
Precompute VAE latents and text embeddings from MDS shards (prepare output).

Works with any prepare script output: datasets/prepare/sstk or datasets/prepare/dreambooth.
MDS input must have columns: image, caption, width, height.

Example:
  python datasets/precompute.py \\
    --datadir ./mds/ \\
    --savedir ./mds_latents_flux2/ \\
    --num_proc 8 \\
    --dataloader_workers 4 \\
    --pretrained_model_name_or_path black-forest-labs/FLUX.2-klein-base-4B \\
    --batch_size 32
"""

import json
import os
import subprocess
import sys
import tempfile
from argparse import ArgumentParser, Namespace
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from streaming import MDSWriter, Stream
from streaming.base.util import merge_index
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from nexus.data.t2i_dataset import StreamingT2IDataset
from nexus.data.utils import text_preprocessing
from nexus.utils import DATA_TYPES


def _datadir_to_streams(datadir: Union[List[str], str]) -> List[Stream]:
    """Convert datadir (str or list of strs) to list of Stream for StreamingDataset."""
    paths = [datadir] if isinstance(datadir, str) else list(datadir)
    return [Stream(local=p) for p in paths]


def build_streaming_sstk_t2i_dataloader(
    datadir: Union[List[str], str],
    batch_size: int,
    resize_sizes: Optional[List[int]] = None,
    drop_last: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    image_key: str = "image",
    caption_key: str = "caption",
    clean_caption: bool = True,
) -> DataLoader:
    assert resize_sizes is not None, "Must provide target resolution for image resizing"

    streams = _datadir_to_streams(datadir)

    transforms_list = [
        transforms.Compose([
            transforms.Resize(
                size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        for size in resize_sizes
    ]

    init_kwargs: dict = {
        "streams": streams,
        "transforms_list": transforms_list,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "image_key": image_key,
        "caption_key": caption_key,
        "clean_caption": clean_caption,
    }
    dataset = StreamingT2IDataset(**init_kwargs)

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
        num_workers=num_workers,
    )

    return dataloader


def _caption_sample_weights(n: int, weights_arg: Optional[List[float]]) -> np.ndarray:
    if weights_arg is None or len(weights_arg) == 0:
        return np.ones(n) / n
    w = np.array(weights_arg[:n], dtype=np.float64)
    if len(w) < n:
        rest = np.ones(n - len(w)) / (n - len(w))
        w = np.concatenate([w, rest])
    return w / w.sum()


def _sample_caption(
    captions: List[str],
    weights: np.ndarray,
    rng: np.random.Generator,
    clean: bool = True,
) -> str:
    idx = rng.choice(len(captions), p=weights)
    return text_preprocessing(captions[idx], clean)[0]


def _discover_subfolders(datadir: str) -> List[str]:
    """Discover worker subfolders (0, 1, 2, ...) from prepare output under datadir."""
    if not os.path.isdir(datadir):
        return []
    subfolders = []
    for name in sorted(os.listdir(datadir)):
        subpath = os.path.join(datadir, name)
        if os.path.isdir(subpath) and name.isdigit():
            subfolders.append(subpath)
    return subfolders


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--datadir", type=str, required=True, help="MDS shards from prepare.")
    parser.add_argument("--savedir", type=str, default="", help="Output path for precomputed latents.")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Workers. If None, one per prepare subfolder.",
    )
    parser.add_argument(
        "--image_resolutions",
        type=int,
        nargs="+",
        default=[512],
    )
    parser.add_argument("--save_images", default=False, action="store_true")
    parser.add_argument(
        "--model_dtype",
        type=str,
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--save_dtype", type=str, choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-4B",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae", default=True, action="store_true")
    parser.add_argument("--no_vae", dest="vae", action="store_false")
    parser.add_argument("--text_encoder", default=True, action="store_true")
    parser.add_argument("--no_text_encoder", dest="text_encoder", action="store_false")
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[9, 18, 27],
    )
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--caption_sample_weights", type=float, nargs="+", default=None)
    parser.add_argument("--dataloader_workers", type=int, default=2)
    parser.add_argument("--worker_idx", type=int, default=None)
    parser.add_argument("--subfolder_paths", type=str, default=None)
    parser.add_argument("--args_file", type=str, default=None)
    args = parser.parse_args()

    if args.args_file is not None:
        with open(args.args_file) as f:
            data = json.load(f)
        args = Namespace(**data)
    if isinstance(args.image_resolutions, int):
        args.image_resolutions = [args.image_resolutions]
    return args


def _precompute_worker(task: Tuple[List[str], int, object]) -> None:
    subfolder_paths, worker_idx, args = task

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        device_idx = worker_idx % num_gpus
        device = torch.device("cuda", device_idx)
        torch.cuda.set_device(device_idx)
    else:
        device = torch.device("cpu")

    torch.manual_seed(worker_idx + args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(worker_idx + args.seed)
    np.random.seed(worker_idx + args.seed)
    rng = np.random.default_rng(worker_idx + args.seed)

    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=DATA_TYPES[args.model_dtype],
    ).to(device).eval()
    vae = torch.compile(vae)

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=DATA_TYPES[args.model_dtype],
    ).to(device).eval()
    text_encoder.requires_grad_(False)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    caption_key = "caption"
    image_key = "image"

    dataloader = build_streaming_sstk_t2i_dataloader(
        datadir=subfolder_paths,
        batch_size=args.batch_size,
        resize_sizes=args.image_resolutions,
        drop_last=False,
        shuffle=False,
        num_workers=args.dataloader_workers,
        image_key=image_key,
        caption_key=caption_key,
        clean_caption=True,
    )
    ds = dataloader.dataset
    n_samples = getattr(ds, "size", None)
    if n_samples is None:
        try:
            n_samples = len(ds)
        except (TypeError, NotImplementedError):
            n_samples = "?"
    print(f"Worker {worker_idx} -> subdirs: {subfolder_paths}, device={device}, samples={n_samples}")

    columns = {"caption": "str"}
    if args.vae:
        for size in args.image_resolutions:
            columns[f"latents_{size}"] = "bytes"
    if args.text_encoder:
        columns["text_embeds"] = "bytes"
    if args.save_images:
        columns["image"] = "jpeg"

    out_dir = os.path.join(args.savedir, str(worker_idx))
    os.makedirs(out_dir, exist_ok=True)
    writer = MDSWriter(
        out=out_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    for batch in tqdm(dataloader, desc=f"Worker {worker_idx}", position=worker_idx):
        images = [
            torch.stack(batch[f"image_{idx}"]).to(device)
            for idx in range(len(args.image_resolutions))
        ]
        batch_size = images[0].shape[0]

        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=DATA_TYPES[args.model_dtype]):
                    latents_dict = {}
                    if args.vae:
                        for idx, size in enumerate(args.image_resolutions):
                            latent_dist = vae.encode(images[idx])
                            assert isinstance(latent_dist, AutoencoderKLOutput)
                            lat = latent_dist.latent_dist.sample().to(
                                DATA_TYPES[args.save_dtype]
                            )
                            latents_dict[size] = lat.detach().cpu().numpy()

                    captions_to_encode = []
                    for i in range(batch_size):
                        c = batch["caption"][i]
                        if isinstance(c, list) and len(c) > 1:
                            weights = _caption_sample_weights(
                                len(c), args.caption_sample_weights
                            )
                            c = _sample_caption(c, weights, rng, clean=True)
                        elif isinstance(c, list) and len(c) == 1:
                            c = text_preprocessing(c[0], True)[0]
                        captions_to_encode.append(c)

                    prompt_embeds = None
                    if args.text_encoder:
                        prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
                            text_encoder=text_encoder,
                            tokenizer=tokenizer,
                            prompt=captions_to_encode,
                            device=device,
                            dtype=DATA_TYPES[args.model_dtype],
                            max_sequence_length=args.max_sequence_length,
                            hidden_states_layers=tuple(args.text_encoder_out_layers),
                        )
                        prompt_embeds = (
                            prompt_embeds.to(DATA_TYPES[args.save_dtype])
                            .detach()
                            .cpu()
                            .numpy()
                        )

                for i in range(batch_size):
                    mds_sample = {"caption": captions_to_encode[i]}
                    if args.text_encoder:
                        mds_sample["text_embeds"] = prompt_embeds[i].tobytes()
                    if args.vae:
                        for size in args.image_resolutions:
                            mds_sample[f"latents_{size}"] = latents_dict[size][i].tobytes()
                    if args.save_images:
                        mds_sample["image"] = batch["sample"][i][image_key]
                    writer.write(mds_sample)
        except RuntimeError as e:
            print(f"Worker {worker_idx} runtime error, skipping batch: {e}")

    writer.finish()

    if torch.cuda.is_available():
        del text_encoder, vae
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    print(f"Worker {worker_idx} finished")


def _partition_subfolders(subfolders: List[str], num_proc: int) -> List[List[str]]:
    if num_proc >= len(subfolders):
        return [[s] for s in subfolders] + [[] for _ in range(num_proc - len(subfolders))]
    chunk_size = (len(subfolders) + num_proc - 1) // num_proc
    return [
        subfolders[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_proc)
        if i * chunk_size < len(subfolders)
    ]


def main(args: ArgumentParser) -> None:
    if args.worker_idx is not None and args.subfolder_paths is not None:
        paths = (
            args.subfolder_paths
            if isinstance(args.subfolder_paths, list)
            else [p.strip() for p in args.subfolder_paths.split(",") if p.strip()]
        )
        _precompute_worker((paths, args.worker_idx, args))
        return

    subfolders = _discover_subfolders(args.datadir)
    num_proc = args.num_proc

    if subfolders:
        if num_proc is None:
            num_proc = len(subfolders)
        num_proc = min(num_proc, len(subfolders))
    else:
        num_proc = 1
        subfolders = [args.datadir]

    partitions = _partition_subfolders(subfolders, num_proc)
    partitions = [p for p in partitions if p]
    num_proc = len(partitions)

    os.makedirs(args.savedir, exist_ok=True)

    procs = []
    script = os.path.abspath(__file__)
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_proc):
            data = {
                k: v
                for k, v in vars(args).items()
                if k not in ("args_file", "worker_idx", "subfolder_paths")
            }
            data["worker_idx"] = i
            data["subfolder_paths"] = partitions[i]
            args_path = os.path.join(tmpdir, f"args_{i}.json")
            with open(args_path, "w") as f:
                json.dump(data, f)
            proc = subprocess.Popen(
                [sys.executable, script, "--args_file", args_path],
                stdout=sys.stdout,
                stderr=sys.stderr,
            )
            procs.append(proc)
        for proc in procs:
            proc.wait()
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)

    shards_metadata = [
        os.path.join(args.savedir, str(i), "index.json") for i in range(num_proc)
    ]
    merge_index(shards_metadata, out=args.savedir, keep_local=True)
    print("Merged all shards into", args.savedir)


if __name__ == "__main__":
    main(parse_args())
