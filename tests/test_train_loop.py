"""Smoke tests for training_step_precomputed (t2i and img2img)."""

import pytest
import torch

from nexus.train.train_loop import training_step_precomputed


@pytest.mark.integration
def test_training_step_img2img_smoke():
    """One img2img step completes and returns finite loss (mocked, no model download)."""
    pytest.importorskip("diffusers")

    from unittest.mock import MagicMock

    from diffusers import Flux2KleinPipeline

    # Minimal shapes - no real model load
    bsz, c, h, w = 2, 32, 8, 8
    latents = torch.randn(bsz, c, h, w)
    source_latents = torch.randn(bsz, c, h, w)

    transformer = MagicMock()
    model_input = Flux2KleinPipeline._patchify_latents(latents)
    seq_len = model_input.size(1)
    dim = model_input.size(-1)

    def forward_fn(*, hidden_states, timestep, guidance, encoder_hidden_states, txt_ids, img_ids, return_dict):
        # For img2img, hidden_states is [noisy_target; source], we return target seq only
        out_seq = hidden_states.size(1)  # full len
        return (torch.randn(bsz, out_seq, dim * 4, dtype=hidden_states.dtype),)

    transformer.forward = forward_fn
    transformer.config.guidance_embeds = False

    noise_scheduler = MagicMock()
    noise_scheduler.config.num_train_timesteps = 1000
    noise_scheduler.timesteps = torch.linspace(0, 999, 1000).long()

    batch = {
        "latents": latents,
        "source_latents": source_latents,
        "text_embeds": torch.randn(bsz, 512, 7680),
        "text_ids": torch.zeros(bsz, 512, 4, dtype=torch.long),
    }
    latents_bn_mean = torch.zeros(dim)
    latents_bn_std = torch.ones(dim)
    accelerator = MagicMock(device=torch.device("cpu"))

    def loss_fn(ctx):
        return (ctx.model_output.mean(), {"loss": ctx.model_output.mean().item()})

    loss, logs = training_step_precomputed(
        batch=batch,
        transformer=transformer,
        latents_bn_mean=latents_bn_mean,
        latents_bn_std=latents_bn_std,
        noise_scheduler_copy=noise_scheduler,
        accelerator=accelerator,
        loss_fn=loss_fn,
    )
    assert torch.isfinite(loss).all()
    assert "loss" in logs
