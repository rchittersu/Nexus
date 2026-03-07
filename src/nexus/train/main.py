"""
Flux.2 Klein training on precomputed MDS data.

Config-driven via YAML. Entry point for LoRA or full fine-tuning of Flux.2
transformers on precomputed VAE latents and text embeddings.

Usage:
    accelerate launch -m nexus.train --config configs/klein4b/t2i_finetune.yaml
    accelerate launch -m nexus.train --config configs/klein4b/t2i_finetune.yaml --output_dir ./out
"""

import warnings

warnings.filterwarnings("ignore", module="distutils")

import copy
import logging
import math
import os
import shutil
from pathlib import Path

import diffusers
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from peft import LoraConfig
from tqdm.auto import tqdm

from streaming import StreamingDataLoader

from nexus.utils.checkpoint_utils import prune_old_checkpoints
from nexus.utils.config_utils import check_prior_preservation_config
from nexus.utils.log_utils import init_trackers, log_dataset_input, setup_tracking

from .config import ns_to_kwargs, parse_args
from nexus.losses import build_loss_fn
from .train_loop import training_step_precomputed

check_min_version("0.37.0.dev0")

# Pipeline class -> (vae_cls, vae_subfolder), (scheduler_cls, scheduler_subfolder)
# Used when loading everything from pipeline except DiT.
_PIPELINE_COMPONENTS = {}
try:
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler

    _PIPELINE_COMPONENTS[Flux2KleinPipeline] = {
        "vae": (AutoencoderKLFlux2, "vae"),
        "scheduler": (FlowMatchEulerDiscreteScheduler, "scheduler"),
    }
except ImportError:
    pass
logger = get_logger(__name__)


def main(args=None):
    """Run Flux.2 Klein LoRA/full training on precomputed MDS data."""
    cfg = parse_args(args)

    # --- Config & logging ---
    config_path = getattr(cfg, "_config_path", None)

    train_cfg = cfg.train
    model_cfg = cfg.model
    lora_cfg = getattr(cfg, "lora", None)
    train_mode = getattr(cfg, "train_mode", "lora")
    pipeline_cfg = getattr(cfg, "pipeline", None) or getattr(model_cfg, "pipeline", None)
    
    # basic sanity checks
    if train_mode == "lora" and not lora_cfg:
        raise ValueError("lora config is required for lora training")
    if not pipeline_cfg:
        raise ValueError("pipeline config is required")
    pretrained_path = getattr(pipeline_cfg, "pretrained_model_name_or_path", None) or getattr(
        model_cfg, "pretrained_model_name_or_path", None
    )
    if not pretrained_path:
        raise ValueError("pipeline.pretrained_model_name_or_path is required")

    # --- Accelerator & trackers ---
    output_dir, log_with = setup_tracking(cfg)

    proj_config = ProjectConfiguration(project_dir=output_dir)
    
    # Make it true if you run into issues with the default value
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        mixed_precision=getattr(cfg, "mixed_precision", None),
        log_with=log_with,
        project_config=proj_config,
        kwargs_handlers=[ddp_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_main_process:
        logger.info("Config: %s", config_path or "(unknown)")
        logger.info(str(accelerator.state))
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if getattr(cfg, "seed", None) is not None:
        set_seed(cfg.seed)

    # --- Model loading ---
    weight_dtype = torch.float32
    mp = getattr(cfg, "mixed_precision", None)
    if mp == "fp16":
        weight_dtype = torch.float16
    elif mp == "bf16":
        weight_dtype = torch.bfloat16


    pipeline_cls = pipeline_cfg._class
    components = _PIPELINE_COMPONENTS.get(pipeline_cls)
    if not components:
        raise ValueError(
            f"No pipeline component registry for {pipeline_cls}. "
            "Add vae/scheduler to model config or extend _PIPELINE_COMPONENTS."
        )

    # Load vae temporarily to get bn stats
    vae_cls, vae_subfolder = components["vae"]
    vae = vae_cls.from_pretrained(
        pretrained_path,
        subfolder=vae_subfolder,
    )
    latents_bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(accelerator.device)
    latents_bn_std = torch.sqrt(
        vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps
    ).to(accelerator.device)
    del vae

    # Load Noise Scheduler
    sched_cls, sched_subfolder = components["scheduler"]
    noise_scheduler = sched_cls.from_pretrained(
        pretrained_path,
        subfolder=sched_subfolder,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Load DiT 
    dit_cfg = getattr(model_cfg, "dit", model_cfg.transformer)
    trans_cls = dit_cfg._class
    subfolder = dit_cfg.subfolder

    transformer = trans_cls.from_pretrained(
        pretrained_path,
        subfolder=subfolder,
    )

    # Carefully Enable the trainable parameters
    transformer.requires_grad_(False)

    lora_config = None
    if train_mode == "lora" and lora_cfg:

        # Assume that the target modules is a list of strings
        target_modules = [m.strip() for m in lora_cfg.target_modules]

        # Create the LoRA config
        lora_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )

        # Add the LoRA config to the transformer
        transformer.add_adapter(lora_config)
    elif train_mode == "full":
        # Enable all the parameters
        transformer.requires_grad_(True)

    # TODO: Check if this is correct
    pipeline_cls = pipeline_cfg._class if train_mode == "lora" else None


    # Enable if there's a memory crunch.
    if train_cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    os.makedirs(cfg.output_dir, exist_ok=True)
    if config_path and accelerator.is_local_main_process:
        dest = Path(cfg.output_dir) / "config.yaml"
        shutil.copy2(config_path, dest)
        logger.info("Config copied to %s", dest)
    is_fsdp = getattr(accelerator.state, "fsdp_plugin", None) is not None
    if train_mode == "lora" and is_fsdp:
        from peft.utils.other import fsdp_auto_wrap_policy

        fsdp_plugin = accelerator.state.fsdp_plugin
        fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy(transformer)

    # TODO: Experiment with this
    if getattr(cfg, "allow_tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    learning_rate = train_cfg.learning_rate
    if train_cfg.scale_lr:
        learning_rate *= (
            train_cfg.gradient_accumulation_steps * train_cfg.batch_size * accelerator.num_processes
        )

    opt_cls = cfg.optimizer._class
    opt_kwargs = ns_to_kwargs(getattr(cfg.optimizer, "kwargs", None))
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = opt_cls(
        trainable_params,
        lr=learning_rate,
        **opt_kwargs,
    )

    # --- Dataset & dataloader ---
    ds_kwargs = ns_to_kwargs(
        cfg.dataset.kwargs,
        batch_size=train_cfg.batch_size,
        latent_dtype=weight_dtype,
    )
   
    train_dataset = cfg.dataset._class(**ds_kwargs)

    # Epoch size must be divisible by num_processes so each process gets the same number of samples.
    epoch_size = getattr(train_dataset, "epoch_size", None)
    if epoch_size is None and callable(getattr(train_dataset, "size", None)):
        epoch_size = train_dataset.size()
    if epoch_size is None:
        epoch_size = len(train_dataset) * accelerator.num_processes
    if epoch_size % accelerator.num_processes != 0:
        raise ValueError(
            f"Dataset epoch size ({epoch_size}) must be divisible by num_processes ({accelerator.num_processes}). "
            "Adjust dataset size or number of processes."
        )

    collate_fn = cfg.collate._fn if hasattr(cfg.collate, "_fn") else None
    if collate_fn is None:
        raise ValueError("collate_fn is required")

    train_dataloader = StreamingDataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=collate_fn,
        num_workers=train_cfg.dataloader_num_workers,
        persistent_workers=train_cfg.dataloader_num_workers > 0,
    )

    num_warmup = train_cfg.lr_warmup_steps * accelerator.num_processes
    # StreamingDataLoader is not wrapped by Accelerate; len() is already per-process.
    len_dl = len(train_dataloader)
    num_updates_per_epoch = math.ceil(len_dl / train_cfg.gradient_accumulation_steps)
    max_steps_cfg = getattr(train_cfg, "max_steps", None)
    # When max_steps not set: default to 1 epoch
    num_training_steps = (
        num_updates_per_epoch if max_steps_cfg is None else max_steps_cfg
    )

    lr_scheduler = get_scheduler(
        train_cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup,
        num_training_steps=num_training_steps,
        num_cycles=train_cfg.lr_num_cycles,
        power=train_cfg.lr_power,
    )

    # Do not wrap the DataLoader with Accelerate; Streaming is ready for distributed out of the box.
    transformer, optimizer, lr_scheduler = accelerator.prepare(
        transformer, optimizer, lr_scheduler
    )

    len_dl_per_process = len_dl
    max_steps = (
        num_updates_per_epoch if max_steps_cfg is None else max_steps_cfg
    )
    num_epochs = math.ceil(max_steps / num_updates_per_epoch)

    # --- Training loop math (printed once, main process only) ---
    total_bs = (
        train_cfg.batch_size
        * accelerator.num_processes
        * train_cfg.gradient_accumulation_steps
    )
    if accelerator.is_main_process:
        logger.info("***** Training loop *****")
        logger.info(
            "  Dataset: len=%s (per process) | batch_size=%s | accumulation=%s | num_proc=%s",
            len_dl_per_process,
            train_cfg.batch_size,
            train_cfg.gradient_accumulation_steps,
            accelerator.num_processes,
        )
        logger.info(
            "  Steps: per_epoch=%s | max_steps=%s | num_epochs=%s | effective_batch_size=%s",
            num_updates_per_epoch,
            max_steps,
            num_epochs,
            total_bs,
        )
        if max_steps_cfg is not None:
            logger.info("  (max_steps=%s set)", max_steps_cfg)

    if accelerator.is_main_process:
        init_trackers(accelerator, cfg)
        log_dataset_input(
            class_name=getattr(cfg.dataset, "class_name", None),
            kwargs=ns_to_kwargs(cfg.dataset.kwargs) if hasattr(cfg.dataset, "kwargs") and cfg.dataset.kwargs else None,
            name=getattr(cfg.dataset, "name", None),
            source_path=ds_kwargs.get("local"),
            context="training",
            resuming=bool(getattr(cfg, "resume_from_checkpoint", None)),
        )

    # --- Loss & validation ---
    loss_cfg = cfg.loss
    loss_fn = build_loss_fn(cfg, accelerator=accelerator, weight_dtype=weight_dtype)
    check_prior_preservation_config(cfg)

    # --- Optional prompt dropout (null text embed) ---
    drop_text_prob = getattr(train_cfg, "drop_text_prob", 0.0)
    null_text_embed_path = getattr(train_cfg, "null_text_embed_path", None)
    if drop_text_prob > 0:
        if not null_text_embed_path:
            raise ValueError(
                "null_text_embed_path must be set in config when drop_text_prob > 0. "
                "Use the empty_string_text_embed notebook to create the .pt file."
            )
        null_text_embeds = torch.load(
            null_text_embed_path, map_location="cpu", weights_only=True
        )
        if null_text_embeds.dim() == 2:
            null_text_embeds = null_text_embeds.unsqueeze(0)
    else:
        null_text_embeds = None

    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info("  Data = %s", ds_kwargs.get("local"))

    global_step = 0
    first_epoch = 0

    path = getattr(cfg, "resume_from_checkpoint", None)
    if path:
        ckpt_path = os.path.join(cfg.output_dir, path)
        accelerator.print(f"Resuming from {path}")
        accelerator.load_state(ckpt_path)
        global_step = int(path.split("-")[1])
        first_epoch = global_step // num_updates_per_epoch
        # StreamingDataLoader is not wrapped by Accelerate; restore its state manually.
        dl_state_path = os.path.join(ckpt_path, "dataloader_state.pt")
        if os.path.exists(dl_state_path):
            dl_state = torch.load(dl_state_path, map_location="cpu", weights_only=True)
            train_dataloader.load_state_dict(dl_state)

    progress_bar = tqdm(
        range(max_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for _ in range(first_epoch, num_epochs):
        transformer.train()
        for _, raw_batch in enumerate(train_dataloader):
            batch = {
                "latents": raw_batch["latents"].to(accelerator.device, dtype=weight_dtype),
                "text_embeds": raw_batch["text_embeds"].to(accelerator.device, dtype=weight_dtype),
                "text_ids": raw_batch["text_ids"].to(accelerator.device),
            }
            if "source_latents" in raw_batch:
                batch["source_latents"] = raw_batch["source_latents"].to(
                    accelerator.device, dtype=weight_dtype
                )

            # Optionally drop prompt (replace with null embed) per sample; text_ids unchanged
            if drop_text_prob > 0 and null_text_embeds is not None:
                B = batch["text_embeds"].shape[0]
                drop_mask = torch.rand(B, device=accelerator.device) < drop_text_prob
                if drop_mask.any():
                    null_emb = null_text_embeds.to(
                        accelerator.device, dtype=weight_dtype
                    ).expand(B, -1, -1)
                    batch["text_embeds"] = torch.where(
                        drop_mask.view(B, 1, 1), null_emb, batch["text_embeds"]
                    )

            with accelerator.accumulate([transformer]):
                loss, loss_breakdown = training_step_precomputed(
                    batch=batch,
                    transformer=transformer,
                    latents_bn_mean=latents_bn_mean,
                    latents_bn_std=latents_bn_std,
                    noise_scheduler_copy=noise_scheduler_copy,
                    accelerator=accelerator,
                    loss_fn=loss_fn,
                    step=global_step,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        transformer.parameters(),
                        cfg.max_grad_norm,
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                logs = {
                    "loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                for k, v in loss_breakdown.items():
                    if k != "loss":
                        logs[k] = v
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if (accelerator.is_main_process or is_fsdp) and global_step % cfg.checkpointing_steps == 0:
                    limit = getattr(cfg, "checkpoints_total_limit", None)
                    if limit is not None:
                        prune_old_checkpoints(cfg.output_dir, limit)
                    save_path = os.path.join(cfg.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    # StreamingDataLoader is not wrapped by Accelerate; save its state manually.
                    if accelerator.is_main_process:
                        dl_state = train_dataloader.state_dict()
                        if dl_state is not None:
                            torch.save(dl_state, os.path.join(save_path, "dataloader_state.pt"))
                    logger.info(f"Saved state to {save_path}")

            if global_step >= max_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
