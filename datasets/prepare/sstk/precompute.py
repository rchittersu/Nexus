import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from streaming import MDSWriter
from streaming.base.util import merge_index
from tqdm import tqdm

from .base import build_streaming_sstk_t2i_dataloader
from src.nexus.utils import DATA_TYPES

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
        default=[10, 20, 30],
        help="Text encoder hidden layers to compute the final text embeddings.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length for text encoding.",
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

    accelerator = Accelerator()
    device = accelerator.device
    device_idx = int(accelerator.process_index)

    # Set random seeds
    torch.manual_seed(device_idx + args.seed)
    torch.cuda.manual_seed(device_idx + args.seed)
    np.random.seed(device_idx + args.seed)

    
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
    dataloader = build_streaming_sstk_t2i_dataloader(
        datadir=[args.datadir],
        batch_size=args.batch_size,
        resize_sizes=args.image_resolutions,
        drop_last=False,
        shuffle=False,
        image_key=image_key,
        caption_key=caption_key,
        clean_caption=True,
        tokenizer=tokenizer,
        prefetch_factor=2,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
    )


    print(f"Device: {device_idx}, Dataloader sample count: {len(dataloader.dataset)}")
    print(
        f"MP variable -> world size: {os.environ['WORLD_SIZE']}, "
        f"RANK: {os.environ['RANK']}, {device}"
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

                    # Encode text with text encoder
                    prompt_embeds = None
                    text_ids = None
                    if args.text_encoder:
                        prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
                            prompt=batch["caption"],  
                            max_sequence_length=args.max_sequence_length,
                            text_encoder_out_layers=args.text_encoder_out_layers,
                        )
                        prompt_embeds = prompt_embeds.to(DATA_TYPES[args.save_dtype]).detach().cpu().numpy()
                        text_ids = text_ids.to(torch.int8).detach().cpu().numpy()

            # Write each sample to MDS
            for i in range(batch_size):
                mds_sample = {
                    "caption": batch["caption"][i],
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