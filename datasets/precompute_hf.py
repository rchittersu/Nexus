"""
Precompute VAE latents and text embeddings from HuggingFace datasets.

Loads datasets directly via load_dataset (no MDS input). Uses Accelerator and
DataLoader for single- or multi-GPU precompute. Supports configurable column
names for prompt, image, and condition (source_image for img2img).

Example (t2i):
  python datasets/precompute_hf.py --dataset username/my-dataset --savedir ./mds_latents/

Example (img2img, multi-GPU):
  accelerate launch datasets/precompute_hf.py --dataset username/my-dataset \\
    --savedir ./mds_latents/ --condition_column source_image
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from streaming import MDSWriter
from streaming.base.util import merge_index
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torchvision import transforms
from tqdm import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from nexus.data.utils import text_preprocessing
from nexus.utils import DATA_TYPES


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


def _ensure_pil(image: Any):
    """Convert HF Image or array to PIL Image if needed."""
    if hasattr(image, "convert"):
        return image.convert("RGB")
    import numpy as np
    from PIL import Image

    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    return Image.fromarray(np.array(image)).convert("RGB")


class HFPrecomputeDataset(Dataset):
    """
    Map-style dataset wrapping a HuggingFace dataset for precompute.

    Yields dicts with image_0, caption, optional source_image_0, and sample.
    """

    def __init__(
        self,
        hf_dataset: Any,
        image_column: str,
        prompt_column: str,
        condition_column: Optional[str],
        resolution: int,
        transform: Callable,
        clean_caption: bool = True,
    ):
        self.hf_dataset = hf_dataset
        self.image_column = image_column
        self.prompt_column = prompt_column
        self.condition_column = condition_column
        self.resolution = resolution
        self.transform = transform
        self.clean_caption = clean_caption

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.hf_dataset[index]
        img = _ensure_pil(row[self.image_column])
        out = {"image_0": self.transform(img)}

        if self.condition_column and self.condition_column in row and row[self.condition_column] is not None:
            cond_img = _ensure_pil(row[self.condition_column])
            out["source_image_0"] = self.transform(cond_img)
        elif self.condition_column:
            # Use target image as fallback when condition missing (img2img with incomplete data)
            out["source_image_0"] = out["image_0"].clone()

        caption = row[self.prompt_column]
        out["caption"] = text_preprocessing(caption, self.clean_caption)
        out["sample"] = row
        return out


class HFPrecomputeIterableDataset(IterableDataset):
    """
    Iterable dataset for streaming HF datasets.
    Shards by (num_processes, process_index) for multi-GPU.
    """

    def __init__(
        self,
        hf_iterable: Any,
        image_column: str,
        prompt_column: str,
        condition_column: Optional[str],
        resolution: int,
        transform: Callable,
        clean_caption: bool = True,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.hf_iterable = hf_iterable
        self.image_column = image_column
        self.prompt_column = prompt_column
        self.condition_column = condition_column
        self.resolution = resolution
        self.transform = transform
        self.clean_caption = clean_caption
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        ds = self.hf_iterable
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        for row in ds:
            img = _ensure_pil(row[self.image_column])
            out = {"image_0": self.transform(img)}

            if self.condition_column and self.condition_column in row and row[self.condition_column] is not None:
                cond_img = _ensure_pil(row[self.condition_column])
                out["source_image_0"] = self.transform(cond_img)
            elif self.condition_column:
                out["source_image_0"] = out["image_0"].clone()

            caption = row[self.prompt_column]
            out["caption"] = text_preprocessing(caption, self.clean_caption)
            out["sample"] = row
            yield out


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset path or local dir")
    parser.add_argument("--dataset_config", type=str, default=None, help="Config name for multi-config datasets")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--image_column", type=str, default="image", help="Column name for target image")
    parser.add_argument("--prompt_column", type=str, default="caption", help="Column name for prompt/caption")
    parser.add_argument(
        "--condition_column",
        type=str,
        default=None,
        help="Optional column for source/condition image (img2img)",
    )
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for large datasets")
    parser.add_argument("--savedir", type=str, default="", help="Output path for precomputed latents")
    parser.add_argument("--resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--save_images", action="store_true")
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
    parser.add_argument("--vae", action="store_true", default=True)
    parser.add_argument("--no_vae", dest="vae", action="store_false")
    parser.add_argument("--text_encoder", action="store_true", default=True)
    parser.add_argument("--no_text_encoder", dest="text_encoder", action="store_false")
    parser.add_argument("--text_encoder_out_layers", type=int, nargs="+", default=[9, 18, 27])
    parser.add_argument("--max_sequence_length", type=int, default=128)
    parser.add_argument("--caption_sample_weights", type=float, nargs="+", default=None)
    parser.add_argument("--dataloader_workers", type=int, default=2)
    return parser.parse_args()


def _custom_collate(batch: List[Dict]) -> Dict:
    out = {k: [] for k in batch[0].keys()}
    for item in batch:
        for key, value in item.items():
            out[key].append(value)
    return out


def main(args: Optional[Namespace] = None) -> None:
    args = args or parse_args()

    accelerator = Accelerator()
    device = accelerator.device

    from accelerate.utils import set_seed
    set_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed + accelerator.process_index)

    # Load HuggingFace dataset
    from datasets import load_dataset

    load_kwargs = {"path": args.dataset, "split": args.split}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    if args.streaming:
        load_kwargs["streaming"] = True

    hf_ds = load_dataset(**load_kwargs)

    # Build transform
    transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Validate columns exist
    if args.streaming:
        # For streaming we can't easily inspect; assume user passed correct cols
        pass
    else:
        cols = hf_ds.column_names
        if args.image_column not in cols:
            raise ValueError(f"image_column '{args.image_column}' not in dataset columns: {cols}")
        if args.prompt_column not in cols:
            raise ValueError(f"prompt_column '{args.prompt_column}' not in dataset columns: {cols}")
        if args.condition_column and args.condition_column not in cols:
            raise ValueError(f"condition_column '{args.condition_column}' not in dataset columns: {cols}")

    has_condition = bool(args.condition_column)

    # Create dataset
    if args.streaming:
        dataset = HFPrecomputeIterableDataset(
            hf_iterable=hf_ds,
            image_column=args.image_column,
            prompt_column=args.prompt_column,
            condition_column=args.condition_column,
            resolution=args.resolution,
            transform=transform,
            clean_caption=True,
            rank=accelerator.process_index,
            world_size=accelerator.num_processes,
        )
    else:
        # DistributedSampler (from accelerator.prepare) handles sharding for map-style
        dataset = HFPrecomputeDataset(
            hf_dataset=hf_ds,
            image_column=args.image_column,
            prompt_column=args.prompt_column,
            condition_column=args.condition_column,
            resolution=args.resolution,
            transform=transform,
            clean_caption=True,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.streaming,
        drop_last=False,
        collate_fn=_custom_collate,
        num_workers=args.dataloader_workers if not args.streaming else 0,
    )
    dataloader = accelerator.prepare(dataloader)

    n_samples = len(dataset) if not args.streaming else "?"
    if accelerator.is_main_process:
        print(f"Precomputing: dataset={args.dataset}, device={device}, samples={n_samples}")

    # Load models
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

    # Output columns
    columns = {"caption": "str"}
    if args.vae:
        columns[f"latents_{args.resolution}"] = "bytes"
    if args.vae and has_condition:
        columns[f"source_latents_{args.resolution}"] = "bytes"
    if args.text_encoder:
        columns["text_embeds"] = "bytes"
    if args.save_images:
        columns["image"] = "jpeg"

    out_dir = os.path.join(args.savedir, str(accelerator.process_index))
    os.makedirs(out_dir, exist_ok=True)
    writer = MDSWriter(
        out=out_dir,
        columns=columns,
        compression=None,
        size_limit=256 * (2**20),
        max_workers=64,
    )

    for batch in tqdm(dataloader, desc=f"Rank {accelerator.process_index}", disable=not accelerator.is_local_main_process):
        images = torch.stack(batch["image_0"]).to(device)
        batch_size = images.shape[0]

        try:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=DATA_TYPES[args.model_dtype]):
                    latents_dict = {}
                    if args.vae:
                        latent_dist = vae.encode(images)
                        assert isinstance(latent_dist, AutoencoderKLOutput)
                        lat = latent_dist.latent_dist.sample().to(DATA_TYPES[args.save_dtype])
                        latents_dict[args.resolution] = lat.detach().cpu().numpy()

                    source_latents_dict = {}
                    if args.vae and has_condition and "source_image_0" in batch:
                        source_images = torch.stack(batch["source_image_0"]).to(device)
                        src_latent_dist = vae.encode(source_images)
                        assert isinstance(src_latent_dist, AutoencoderKLOutput)
                        src_lat = src_latent_dist.latent_dist.sample().to(DATA_TYPES[args.save_dtype])
                        source_latents_dict[args.resolution] = src_lat.detach().cpu().numpy()

                    captions_to_encode = []
                    for i in range(batch_size):
                        c = batch["caption"][i]
                        if isinstance(c, list) and len(c) > 1:
                            weights = _caption_sample_weights(len(c), args.caption_sample_weights)
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
                            prompt_embeds.to(DATA_TYPES[args.save_dtype]).detach().cpu().numpy()
                        )

                for i in range(batch_size):
                    mds_sample = {"caption": captions_to_encode[i]}
                    if args.text_encoder:
                        mds_sample["text_embeds"] = prompt_embeds[i].tobytes()
                    if args.vae:
                        mds_sample[f"latents_{args.resolution}"] = latents_dict[args.resolution][i].tobytes()
                    if args.vae and has_condition and args.resolution in source_latents_dict:
                        mds_sample[f"source_latents_{args.resolution}"] = source_latents_dict[
                            args.resolution
                        ][i].tobytes()
                    if args.save_images:
                        mds_sample["image"] = batch["sample"][i].get(args.image_column)
                    writer.write(mds_sample)
        except RuntimeError as e:
            if accelerator.is_local_main_process:
                print(f"Rank {accelerator.process_index} runtime error, skipping batch: {e}")

    writer.finish()

    if torch.cuda.is_available():
        del text_encoder, vae
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    accelerator.wait_for_everyone()

    # Merge indices (main process only)
    if accelerator.is_main_process and args.savedir:
        shards_meta = [
            os.path.join(args.savedir, str(i), "index.json")
            for i in range(accelerator.num_processes)
        ]
        if all(os.path.exists(p) for p in shards_meta):
            merge_index(shards_meta, out=args.savedir, keep_local=True)
            print("Merged into", args.savedir)

    if accelerator.is_local_main_process:
        print(f"Rank {accelerator.process_index} finished")


if __name__ == "__main__":
    main()
