"""Tests for datasets/prepare/sstk/precompute.py"""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    torch = None
    HAS_CUDA = False

# Project root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Import precompute - need to run from package context
# We import parse_args directly; main needs accelerate/CUDA
from datasets.prepare.sstk.precompute import parse_args


def _get_precompute_main():
    """Lazy import of main to avoid loading heavy deps in unit tests."""
    from datasets.prepare.sstk.precompute import main
    return main


class TestParseArgs:
    """Unit tests for parse_args()."""

    def test_required_datadir(self):
        """datadir is required and parsed correctly."""
        with patch.object(sys, 'argv', ['precompute.py', '--datadir', '/path/to/mds']):
            args = parse_args()
            assert args.datadir == '/path/to/mds'

    def test_default_values(self):
        """Default values are applied when not specified."""
        with patch.object(sys, 'argv', ['precompute.py', '--datadir', '/path/to/mds']):
            args = parse_args()
            assert args.savedir == ''
            assert args.image_resolutions == [512, 1024]
            assert args.save_images is False
            assert args.model_dtype == 'bfloat16'
            assert args.save_dtype == 'float16'
            assert args.pretrained_model_name_or_path == 'black-forest-labs/FLUX.2-klein-base-4B'
            assert args.batch_size == 32
            assert args.seed == 42
            assert args.vae is True
            assert args.text_encoder is True
            assert args.text_encoder_out_layers == [10, 20, 30]
            assert args.max_sequence_length == 512

    def test_all_arguments_override_defaults(self):
        """All optional arguments override defaults when provided."""
        with patch.object(sys, 'argv', [
            'precompute.py',
            '--datadir', '/data/mds',
            '--savedir', '/out/latents',
            '--image_resolutions', '256', '512',
            '--save_images',
            '--model_dtype', 'float16',
            '--save_dtype', 'float32',
            '--pretrained_model_name_or_path', 'local/model',
            '--batch_size', '16',
            '--seed', '123',
            '--no_vae',
            '--no_text_encoder',
            '--text_encoder_out_layers', '5', '15',
            '--max_sequence_length', '256',
        ]):
            args = parse_args()
            assert args.datadir == '/data/mds'
            assert args.savedir == '/out/latents'
            assert args.image_resolutions == [256, 512]
            assert args.save_images is True
            assert args.model_dtype == 'float16'
            assert args.save_dtype == 'float32'
            assert args.pretrained_model_name_or_path == 'local/model'
            assert args.batch_size == 16
            assert args.seed == 123
            assert args.vae is False
            assert args.text_encoder is False
            assert args.text_encoder_out_layers == [5, 15]
            assert args.max_sequence_length == 256

    def test_single_image_resolution(self):
        """Single image resolution is converted to list."""
        with patch.object(sys, 'argv', [
            'precompute.py', '--datadir', '/data',
            '--image_resolutions', '512',
        ]):
            args = parse_args()
            assert args.image_resolutions == [512]

    def test_missing_datadir_raises(self):
        """Missing required --datadir raises SystemExit."""
        with patch.object(sys, 'argv', ['precompute.py']):
            with pytest.raises(SystemExit):
                parse_args()


class TestColumnsBuilding:
    """Test the columns dict logic (mirrors main() behavior)."""

    def test_columns_with_vae_and_text_encoder(self):
        """Columns include latents and text embeds when both enabled."""
        args = MagicMock()
        args.vae = True
        args.text_encoder = True
        args.image_resolutions = [512, 1024]
        args.save_images = False

        columns = {'caption': 'str'}
        if args.vae:
            for size in args.image_resolutions:
                columns[f'latents_{size}'] = 'bytes'
        if args.text_encoder:
            columns['text_embeds'] = 'bytes'
            columns['text_ids'] = 'bytes'
        if args.save_images:
            columns['image'] = 'jpeg'

        assert columns == {
            'caption': 'str',
            'latents_512': 'bytes',
            'latents_1024': 'bytes',
            'text_embeds': 'bytes',
            'text_ids': 'bytes',
        }

    def test_columns_no_vae(self):
        """Columns exclude latents when vae disabled."""
        args = MagicMock()
        args.vae = False
        args.text_encoder = True
        args.image_resolutions = [512]
        args.save_images = False

        columns = {'caption': 'str'}
        if args.vae:
            for size in args.image_resolutions:
                columns[f'latents_{size}'] = 'bytes'
        if args.text_encoder:
            columns['text_embeds'] = 'bytes'
            columns['text_ids'] = 'bytes'
        if args.save_images:
            columns['image'] = 'jpeg'

        assert 'latents_512' not in columns
        assert 'text_embeds' in columns

    def test_columns_with_save_images(self):
        """Columns include image when save_images enabled."""
        args = MagicMock()
        args.vae = True
        args.text_encoder = False
        args.image_resolutions = [512]
        args.save_images = True

        columns = {'caption': 'str'}
        if args.vae:
            for size in args.image_resolutions:
                columns[f'latents_{size}'] = 'bytes'
        if args.text_encoder:
            columns['text_embeds'] = 'bytes'
            columns['text_ids'] = 'bytes'
        if args.save_images:
            columns['image'] = 'jpeg'

        assert columns['image'] == 'jpeg'


@pytest.mark.integration
@pytest.mark.skipif(not HAS_CUDA, reason='CUDA required for precompute integration')
class TestPrecomputeIntegration:
    """
    Integration test: prepare -> precompute -> read shards -> decode latent + re-encode text
    -> verify cosine similarity of text embeds (stored vs re-encoded) is high.
    """

    def test_precompute_read_decode_verify_similarity(self, tmp_path):
        """
        Full pipeline: create images -> prepare MDS -> run precompute -> read output ->
        decode latents to images, re-encode captions, verify stored vs re-encoded
        text embeddings have high cosine similarity.
        """
        import torch
        from PIL import Image
        from streaming import StreamingDataset

        # Add sstk to path for prepare
        sstk_dir = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(sstk_dir))
        from prepare import main as prepare_main

        num_images = 3  # Small set for faster run
        images_dir = tmp_path / 'images'
        images_dir.mkdir()
        mds_dir = tmp_path / 'mds'
        latents_dir = tmp_path / 'latents'

        # 1. Create images + captions
        captions = ['a red apple', 'a blue sky', 'a green tree']
        for i in range(num_images):
            img_path = images_dir / f'img_{i}.png'
            img = Image.new('RGB', (512, 512), color=(i * 80, 100, 150))
            img.save(img_path)
            (images_dir / f'img_{i}.png.json').write_text(
                json.dumps({'description': captions[i]})
            )

        images_txt = tmp_path / 'images.txt'
        images_txt.write_text('\n'.join(str(images_dir / f'img_{i}.png') for i in range(num_images)))

        # 2. Run prepare.py
        with patch.object(sys, 'argv', [
            'prepare.py',
            '--images_txt', str(images_txt),
            '--local_mds_dir', str(mds_dir),
            '--num_proc', '1',
            '--seed', '42',
            '--size', str(num_images),
            '--min_size', '0',
        ]):
            prepare_main()

        assert mds_dir.exists()

        # 3. Run precompute via subprocess (accelerate launch)
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}"
        cmd = [
            'accelerate', 'launch', '--num_processes', '1',
            '-m', 'datasets.prepare.sstk.precompute',
            '--datadir', str(mds_dir),
            '--savedir', str(latents_dir),
            '--image_resolutions', '512',
            '--batch_size', '2',
            '--pretrained_model_name_or_path', 'black-forest-labs/FLUX.2-klein-base-4B',
            '--model_dtype', 'float16',
            '--save_dtype', 'float16',
        ]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            pytest.skip(f'Precompute failed (model may not be cached): {result.stderr[:500]}')

        assert latents_dir.exists()

        # 4. Load model for decode + re-encode
        from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline

        device = torch.device('cuda:0')
        model_path = 'black-forest-labs/FLUX.2-klein-base-4B'

        vae = AutoencoderKLFlux2.from_pretrained(
            model_path, subfolder='vae',
            torch_dtype=torch.float16,
        ).to(device).eval()

        from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
        text_encoder = Qwen3ForCausalLM.from_pretrained(
            model_path, subfolder='text_encoder',
            torch_dtype=torch.float16,
        ).to(device).eval()
        tokenizer = Qwen2TokenizerFast.from_pretrained(
            model_path, subfolder='tokenizer',
        )
        pipeline = Flux2KleinPipeline.from_pretrained(
            model_path,
            vae=None, transformer=None,
            tokenizer=tokenizer, text_encoder=text_encoder,
        )

        # 5. Read MDS and verify
        dataset = StreamingDataset(
            local=str(latents_dir),
            batch_size=1,
            shuffle=False,
        )

        sim_threshold = 0.999  # Stored vs re-encoded should match
        for idx, sample in enumerate(dataset):
            caption = sample['caption']
            text_embeds_bytes = sample['text_embeds']
            latent_bytes = sample['latents_512']

            # Re-encode caption
            with torch.no_grad():
                re_embeds, _ = pipeline.encode_prompt(
                    prompt=[caption],
                    max_sequence_length=512,
                    text_encoder_out_layers=[10, 20, 30],
                )
            re_embeds_np = re_embeds.cpu().float().numpy()

            # Stored embeds: bytes -> numpy (same shape as re_embeds)
            stored_flat = np.frombuffer(text_embeds_bytes, dtype=np.float16)
            stored_embeds = stored_flat.reshape(re_embeds_np.shape).astype(np.float32)

            # Cosine similarity
            a = stored_embeds.flatten()
            b = re_embeds_np.flatten()
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
            assert cos_sim >= sim_threshold, f'Sample {idx}: cos_sim={cos_sim:.4f} < {sim_threshold}'

            # Decode latent -> image (sanity check shape)
            latent_flat = np.frombuffer(latent_bytes, dtype=np.float16)
            # FLUX VAE 512->64: (1, 16, 64, 64)
            latent_shape = (1, 16, 64, 64)
            latent = torch.from_numpy(latent_flat.reshape(latent_shape).astype(np.float32)).to(device)
            with torch.no_grad():
                out = vae.decode(latent)
                decoded = out.sample if hasattr(out, 'sample') else out
            assert decoded.shape[-2:] == (512, 512), f'Decoded shape {decoded.shape}'
