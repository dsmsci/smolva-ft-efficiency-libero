from __future__ import annotations

import gc
import random
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .settings import DATASET_RENAME_MAP, OUTPUTS, VIDEO_BACKEND
from .utils import seed_everything


def unwrap_policy(policy):
    return policy


def temporal_dataset(
    repo_id: str,
    root: Path,
    policy_config,
    *,
    episodes: list[int] | None = None,
    video_horizon: int | None = None,
):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta: dict[str, list[float]] = {
        "action": [i / meta.fps for i in policy_config.action_delta_indices]
    }
    if video_horizon is not None:
        for key in meta.video_keys:
            delta[key] = [0.0, video_horizon / meta.fps]

    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta,
        video_backend=VIDEO_BACKEND,
    )
    if episodes is not None:
        loaded = [int(x) for x in dataset.episodes]
        if loaded != [int(x) for x in episodes]:
            raise RuntimeError(f"Loaded episodes {loaded}, requested {episodes}")
    return dataset


def make_loader(
    dataset,
    batch_size: int,
    seed: int,
    *,
    start_step: int = 0,
    drop_n_last_frames: int = 0,
):
    from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
    from lerobot.utils.collate import lerobot_collate_fn

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=drop_n_last_frames,
        shuffle=True,
        seed=seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    if start_step:
        sampler.load_state_dict(compute_sampler_state(start_step, len(sampler), batch_size, 1))

    collate = lerobot_collate_fn if dataset.meta.has_language_columns else None
    # Worker seeding must not consume the global torch RNG used by the model.
    # This also makes custom-loop resume independent of DataLoader iterator creation.
    loader_generator = torch.Generator().manual_seed(int(seed) + 1_000_003)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        multiprocessing_context="spawn",
        drop_last=False,
        collate_fn=collate,
        generator=loader_generator,
    )


def load_policy_and_processors(checkpoint: str | Path, dataset):
    """Mirror LeRobot 0.6.1 training: load checkpoint, but use target dataset stats."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = Path(checkpoint)
    config.device = "cuda"
    config.use_amp = False
    config.empty_cameras = 1

    policy = make_policy(cfg=config, ds_meta=dataset.meta, rename_map=DATASET_RENAME_MAP)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": DATASET_RENAME_MAP},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        },
    )
    return policy, preprocessor, postprocessor


def make_preprocessor(
    checkpoint: str | Path,
    policy,
    dataset,
    *,
    normalization_stats: dict | None = None,
):
    """Build a training preprocessor, optionally forcing one shared normalization space.

    ``dataset`` determines the raw feature schema. ``normalization_stats`` may come
    from another compatible LIBERO view; H4 uses the target K-only statistics for
    both target and seen replay batches so the action head is not trained in two
    different normalized coordinate systems.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = Path(checkpoint)
    config.device = "cuda"
    config.use_amp = False
    stats = dataset.meta.stats if normalization_stats is None else normalization_stats
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        dataset_stats=stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": DATASET_RENAME_MAP},
            "normalizer_processor": {
                "stats": stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        },
    )
    return preprocessor


def configure_optimizer(policy, steps: int, extra_parameters: Iterable[torch.nn.Parameter] = ()):
    base = unwrap_policy(policy)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    trainable.extend([p for p in extra_parameters if p.requires_grad])
    if not trainable:
        raise RuntimeError("No trainable parameters")

    optimizer_config = base.config.get_optimizer_preset()
    if hasattr(base.config, "scheduler_decay_steps"):
        base.config.scheduler_decay_steps = int(steps)
    if hasattr(base.config, "scheduler_warmup_steps"):
        base.config.scheduler_warmup_steps = min(1000, max(100, steps // 10))
    scheduler_config = base.config.get_scheduler_preset()
    optimizer = optimizer_config.build(trainable)
    scheduler = scheduler_config.build(optimizer, num_training_steps=steps)
    return trainable, optimizer, scheduler, float(optimizer_config.grad_clip_norm)


def _prepare_raw_images(batch: dict) -> dict:
    """Match LeRobot training: uint8 RGB tensors become float tensors in [0,1]."""
    prepared = dict(batch)
    for key, value in list(prepared.items()):
        if key.startswith("observation.images.") and torch.is_tensor(value) and value.dtype == torch.uint8:
            prepared[key] = value.to(torch.float32).div_(255.0)
    return prepared


def next_batch(iterator, loader):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return _prepare_raw_images(batch), iterator


def select_video_time(raw: dict, index: int) -> dict:
    """Select t or t+h from two-frame video tensors without touching action chunks."""
    selected = dict(raw)
    for key, value in list(selected.items()):
        if not hasattr(value, "ndim") or value.ndim < 2 or value.shape[1] != 2:
            continue
        if key.startswith("observation.images.") or key.endswith("_is_pad"):
            selected[key] = value[:, index]
    return selected


def visual_embedding(policy, processed: dict) -> torch.Tensor:
    """Frozen image-only representation; language/state are intentionally not used."""
    base = unwrap_policy(policy)
    images, image_masks = base.prepare_images(processed)
    pooled = []
    weights = []
    for image, mask in zip(images, image_masks, strict=True):
        tokens = base.model.vlm_with_expert.embed_image(image)
        pooled.append(tokens.mean(dim=1))
        weights.append(mask.to(tokens.dtype).reshape(-1, 1))
    z = torch.stack(pooled, dim=1)
    w = torch.stack(weights, dim=1)
    return (z * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def language_embedding(policy, processed: dict) -> torch.Tensor:
    base = unwrap_policy(policy)
    tokens = processed["observation.language.tokens"]
    mask = processed["observation.language.attention_mask"].bool()
    embedded = base.model.vlm_with_expert.embed_language_tokens(tokens)
    weight = mask.to(embedded.dtype).unsqueeze(-1)
    return (embedded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


def policy_loss_with_action_hidden(policy, processed: dict) -> tuple[torch.Tensor, torch.Tensor]:
    base = unwrap_policy(policy)
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, inputs):
        if not inputs:
            raise RuntimeError("action_out_proj pre-hook received no input")
        captured["hidden"] = inputs[0]

    handle = base.model.action_out_proj.register_forward_pre_hook(capture)
    try:
        loss, _ = policy(processed)
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("Could not capture SmolVLA action-expert hidden states")
    return loss, captured["hidden"]


def save_resume(
    run_root: Path,
    step: int,
    policy,
    optimizer,
    scheduler,
    *,
    aux_module: torch.nn.Module | None = None,
) -> None:
    path = run_root / "checkpoints" / "last.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = {
        "step": int(step),
        "trainable_state": {
            name: parameter.detach().cpu()
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }
    if aux_module is not None:
        payload["aux_state"] = {k: v.detach().cpu() for k, v in aux_module.state_dict().items()}
    torch.save(payload, tmp)
    tmp.replace(path)


def restore_resume(
    run_root: Path,
    policy,
    optimizer,
    scheduler,
    *,
    aux_module: torch.nn.Module | None = None,
) -> int:
    path = run_root / "checkpoints" / "last.pt"
    if not path.exists():
        return 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    parameters = dict(policy.named_parameters())
    for name, value in state["trainable_state"].items():
        if name not in parameters:
            raise RuntimeError(f"Resume parameter not found: {name}")
        parameters[name].data.copy_(value.to(parameters[name].device, dtype=parameters[name].dtype))
    if aux_module is not None and "aux_state" in state:
        aux_module.load_state_dict(state["aux_state"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    print(f"resumed {run_root} from step {state['step']}")
    return int(state["step"])


def save_final(run_root: Path, policy, preprocessor, postprocessor) -> Path:
    final = run_root / "final"
    if final.exists():
        return final
    tmp = run_root / ".final_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(tmp)
    policy.config.save_pretrained(tmp)
    preprocessor.save_pretrained(tmp)
    postprocessor.save_pretrained(tmp)
    tmp.replace(final)
    shutil.rmtree(run_root / "checkpoints", ignore_errors=True)
    return final


def setup_training(seed: int) -> None:
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


def cleanup_cuda(*objects) -> None:
    del objects
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
