import os
import time
from argparse import ArgumentParser
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator
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
    image_key: str = 'image',
    caption_key: str = 'caption',
    clean_caption: bool = True,
) -> DataLoader:
    assert resize_sizes is not None, 'Must provide target resolution for image resizing'

    streams = _datadir_to_streams(datadir)

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

    # num_workers=0 required: StreamingDataset + multiprocessing workers causes deadlock with DDP
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=custom_collate,
        num_workers=0,
    )

    return dataloader

def _caption_sample_weights(n: int, weights_arg: Optional[List[float]]) -> np.ndarray:
    """Build probability weights for n caption indices. If weights_arg has fewer
    entries, remaining indices get equal share. Normalized to sum to 1."""
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
    """Sample one caption by index, preprocess, return string."""
    idx = rng.choice(len(captions), p=weights)
    chosen = captions[idx]
    return text_preprocessing(chosen, clean)[0]


"""Example usage:
accelerate launch --multi_gpu --num_processes 8 precompute.py \
    --datadir ./sa1b/mds/ \
    --savedir ./sa1b/mds_latents_sdxl1_dfnclipH14/ \
    --pretrained_model_name_or_path <path_to_pretrained_model> \
    --batch_size 32
"""


def parse_args() -> ArgumentParser:
    """Parse command-line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--datadir",
        type=str,
        required=True,
        help="Local directory to store mds shards.",
    )
    parser.add_argument(
        "--savedir",
        type=str,
        default="",
        help="Remote path to upload MDS-formatted shards to.",
    )
    parser.add_argument(
        "--image_resolutions",
        type=int,
        nargs="+",
        default=[512, 1024],
        help="List of image resolutions to use for processing.",
    )
    parser.add_argument(
        "--save_images",
        default=False,
        action="store_true",
        help="If True, also save images, else only latents",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Data type for the encoding models",
    )
    parser.add_argument(
        "--save_dtype",
        type=str,
        choices=("float16", "float32"),
        default="float16",
        help="Data type to save the latents",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-base-4B",
        help="Path to pretrained model.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size per device to use for encoding.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random number generation.",
    )
    parser.add_argument(
        "--vae",
        default=True,
        action="store_true",
        help="If True, encode images with VAE and save latents.",
    )
    parser.add_argument(
        "--no_vae",
        dest="vae",
        action="store_false",
        help="Disable VAE encoding.",
    )
    parser.add_argument(
        "--text_encoder",
        default=True,
        action="store_true",
        help="If True, encode captions with text encoder and save embeddings.",
    )
    parser.add_argument(
        "--no_text_encoder",
        dest="text_encoder",
        action="store_false",
        help="Disable text encoder encoding.",
    )
    parser.add_argument(
        "--text_encoder_out_layers",
        type=int,
        nargs="+",
        default=[9, 18, 27],
        help="Text encoder hidden layers to compute the final text embeddings.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length for text encoding.",
    )
    parser.add_argument(
        "--caption_sample_weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for sampling among multiple captions by index (e.g. 0.5,0.3,0.2). "
        "If fewer weights than captions, remaining indices get equal weight. Default: uniform.",
    )
    args = parser.parse_args()
    if isinstance(args.image_resolutions, int):
        args.image_resolutions = [args.image_resolutions]
    return args


def main(args: ArgumentParser) -> None:
    """Precompute image and text latents and store them in MDS format.

    By default, saves image latents for 512x512 and 1024x1024 resolutions
    (using center crop), and text embeddings from the Qwen text encoder.

    Note that the image latents will be scaled by the vae_scaling_factor.
    """

    # Set device before distributed init to avoid "using gpu x to perform barrier" warning
    if torch.cuda.is_available() and "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    accelerator = Accelerator()
    device = accelerator.device
    device_idx = int(accelerator.process_index)

    # Set random seeds
    torch.manual_seed(device_idx + args.seed)
    torch.cuda.manual_seed(device_idx + args.seed)
    np.random.seed(device_idx + args.seed)
    rng = np.random.default_rng(device_idx + args.seed)

    
    # load text encoder and vae
    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",  # Change subfolder if needed
        torch_dtype=DATA_TYPES[args.model_dtype],
    ).to(device).eval()
    vae = torch.compile(vae)

    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        torch_dtype=DATA_TYPES[args.model_dtype],
    )
    text_encoder.requires_grad_(False)
    text_encoder.to(device)

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
    )

    # initialise text encoding pipeline
    text_encoding_pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=None,
        transformer=None,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )

    caption_key = "caption"
    image_key = "image"
    # Ensure RANK/WORLD_SIZE are set for StreamingDataset partitioning (accelerate sets these,
    # but we ensure they match our process state)
    if accelerator.num_processes > 1:
        os.environ["RANK"] = str(accelerator.process_index)
        os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
    dataloader = build_streaming_sstk_t2i_dataloader(
        datadir=args.datadir,
        batch_size=args.batch_size,
        resize_sizes=args.image_resolutions,
        drop_last=False,
        shuffle=False,
        image_key=image_key,
        caption_key=caption_key,
        clean_caption=True,
    )
    # Avoid accelerator.prepare(dataloader): StreamingDataset handles distributed via RANK/WORLD_SIZE
    # env vars. prepare() injects DistributedSampler which conflicts and causes deadlock (see
    # mosaicml/streaming#307). Only prepare when single-process.
    if accelerator.num_processes == 1:
        dataloader = accelerator.prepare(dataloader)


    ds = dataloader.dataset
    n_samples = getattr(ds, "size", None)
    if n_samples is None:
        try:
            n_samples = len(ds)
        except (TypeError, NotImplementedError):
            n_samples = "?"
    print(
        f"Device: {device_idx}, world size: {accelerator.num_processes}, "
        f"dataset samples: {n_samples}, {device}"
    )

    columns = {'caption': 'str'}
    if args.vae:
        for size in args.image_resolutions:
            columns[f'latents_{size}'] = 'bytes'
    if args.text_encoder:
        columns['text_embeds'] = 'bytes'
        columns['text_ids'] = 'bytes'
    if args.save_images:
        columns["image"] = 'jpeg'

    remote_upload = os.path.join(args.savedir, str(accelerator.process_index))
    writer = MDSWriter(
        out=remote_upload,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )


    for batch in tqdm(dataloader):
        # Stack images for each resolution
        images = [
            torch.stack(batch[f"image_{idx}"]).to(device)
            for idx in range(len(args.image_resolutions))
        ]

        batch_size = images[0].shape[0]

        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=DATA_TYPES[args.model_dtype]):
                    # Encode images with VAE for each resolution
                    latents_dict = {}
                    if args.vae:
                        for idx, size in enumerate(args.image_resolutions):
                            latent_dist = vae.encode(images[idx])
                            assert isinstance(latent_dist, AutoencoderKLOutput)
                            lat = (
                                latent_dist.latent_dist.sample()
                            ).to(DATA_TYPES[args.save_dtype])
                            latents_dict[size] = lat.detach().cpu().numpy()

                    # Sample one caption per sample when multiple exist, then encode
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
                        # else c is already a string
                        captions_to_encode.append(c)

                    # Encode text with text encoder
                    prompt_embeds = None
                    text_ids = None
                    if args.text_encoder:
                        prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                            prompt=captions_to_encode,
                            max_sequence_length=args.max_sequence_length,
                            text_encoder_out_layers=args.text_encoder_out_layers,
                        )
                        prompt_embeds = prompt_embeds.to(DATA_TYPES[args.save_dtype]).detach().cpu().numpy()
                        text_ids = text_ids.to(torch.int8).detach().cpu().numpy()

            # Write each sample to MDS
            for i in range(batch_size):
                mds_sample = {
                    "caption": captions_to_encode[i],
                }
                if args.text_encoder:
                    mds_sample["text_embeds"] = prompt_embeds[i].tobytes()
                    mds_sample["text_ids"] = text_ids[i].tobytes()
                if args.vae:
                    for size in args.image_resolutions:
                        mds_sample[f"latents_{size}"] = latents_dict[size][i].tobytes()
                if args.save_images:
                    mds_sample["image"] = batch["sample"][i][image_key]
                writer.write(mds_sample)
        except RuntimeError as e:
            print(f"Runtime error CUDA, skipping this batch: {e}")

    writer.finish()

    # Wait for all processes to finish
    accelerator.wait_for_everyone()
    print(f"Process {accelerator.process_index} finished")
    time.sleep(10)

    # Merge the mds shards created by each device (only do on main process)
    if accelerator.is_main_process:
        shards_metadata = [
            os.path.join(args.savedir, str(i), "index.json")
            for i in range(accelerator.num_processes)
        ]
        merge_index(shards_metadata, out=args.savedir, keep_local=True)


if __name__ == "__main__":
    main(parse_args())