from __future__ import annotations

import copy
import gc
import shutil
from dataclasses import asdict
from pathlib import Path

import torch

from .data import subset_root, target_repo_id
from .models import (
    DynamicsConfig,
    LatentDynamicsHead,
    dynamics_loss,
    flow_clean_action,
    prefix_latent,
    unwrap_policy,
)
from .settings import (
    BUDGETS,
    DATASET_RENAME_MAP,
    OUTPUTS,
    SEEN_REPO_ID,
    SEEN_TRAIN_ROOT,
    TARGETS,
    TARGET_SEEDS,
)


def _dataloader(dataset, batch_size: int, seed: int, start_step: int = 0):
    from lerobot.datasets import EpisodeAwareSampler, compute_sampler_state
    from lerobot.utils.collate import lerobot_collate_fn

    sampler = EpisodeAwareSampler(
        dataset.meta.episodes["dataset_from_index"],
        dataset.meta.episodes["dataset_to_index"],
        episode_indices_to_use=dataset.episodes,
        drop_n_last_frames=0,
        shuffle=True,
        seed=seed,
        absolute_to_relative_idx=dataset.absolute_to_relative_idx,
    )
    if start_step:
        sampler.load_state_dict(
            compute_sampler_state(start_step, len(sampler), batch_size, 1)
        )

    collate = lerobot_collate_fn if dataset.meta.has_language_columns else None
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        multiprocessing_context="spawn",
        drop_last=False,
        collate_fn=collate,
    )


def _policy_optimizer(policy, steps: int):
    base = unwrap_policy(policy)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer_config = base.config.get_optimizer_preset()
    if hasattr(base.config, "scheduler_decay_steps"):
        base.config.scheduler_decay_steps = int(steps)
    if hasattr(base.config, "scheduler_warmup_steps"):
        base.config.scheduler_warmup_steps = min(1000, max(100, steps // 10))
    scheduler_config = base.config.get_scheduler_preset()
    optimizer = optimizer_config.build(trainable)
    scheduler = scheduler_config.build(optimizer, num_training_steps=steps)
    return trainable, optimizer, scheduler, float(optimizer_config.grad_clip_norm)


def _load_policy(checkpoint: str | Path, dataset, *, lora: bool):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = Path(checkpoint)
    config.device = "cuda"
    config.use_amp = False
    config.empty_cameras = 1

    policy = make_policy(
        cfg=config,
        ds_meta=dataset.meta,
        rename_map=DATASET_RENAME_MAP,
    )
    if lora:
        policy = policy.wrap_with_peft(
            peft_cli_overrides={
                "method_type": "LORA",
                "r": 64,
                "lora_alpha": 64,
            }
        )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {
                "rename_map": DATASET_RENAME_MAP,
            },
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {
                    **policy.config.input_features,
                    **policy.config.output_features,
                },
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


def _seen_preprocessor(seen_checkpoint: str | Path, policy):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.policies import make_pre_post_processors

    config = PreTrainedConfig.from_pretrained(seen_checkpoint)
    config.pretrained_path = Path(seen_checkpoint)
    config.device = "cuda"
    config.use_amp = False

    seen_meta = LeRobotDatasetMetadata(SEEN_REPO_ID, root=SEEN_TRAIN_ROOT)
    base = unwrap_policy(policy)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(seen_checkpoint),
        dataset_stats=seen_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {
                "rename_map": DATASET_RENAME_MAP,
            },
            "normalizer_processor": {
                "stats": seen_meta.stats,
                "features": {
                    **base.config.input_features,
                    **base.config.output_features,
                },
                "norm_map": base.config.normalization_mapping,
            },
        },
    )
    return preprocessor


def _temporal_dataset(repo_id: str, root: Path, policy_config, horizon: int):
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    delta = {
        key: [0, horizon / meta.fps]
        for key in meta.features
        if key.startswith("observation.")
    }
    delta["action"] = [
        index / meta.fps
        for index in policy_config.action_delta_indices
    ]
    return LeRobotDataset(repo_id, root=root, delta_timestamps=delta)


def _observation_at(raw: dict, index: int) -> dict:
    selected = copy.deepcopy(raw)
    for key, value in list(selected.items()):
        if (
            key.startswith("observation.")
            and hasattr(value, "shape")
            and value.ndim >= 2
            and value.shape[1] == 2
        ):
            selected[key] = value[:, index]
    return selected


def _future_valid(raw: dict, device):
    batch = next(
        value.shape[0]
        for value in raw.values()
        if hasattr(value, "shape") and value.ndim > 0
    )
    valid = torch.ones(batch, dtype=torch.bool, device=device)
    found_padding = False
    for key, value in raw.items():
        if (
            key.startswith("observation.")
            and key.endswith("_is_pad")
            and hasattr(value, "shape")
            and value.ndim >= 2
            and value.shape[1] == 2
        ):
            valid &= ~value[:, 1].to(device).bool()
            found_padding = True
    return valid if found_padding else None


def _save_state(
    run_root: Path,
    step: int,
    policy,
    optimizer,
    scheduler,
) -> None:
    path = run_root / "checkpoints" / "last.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "trainable_state": {
                name: parameter.detach().cpu()
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        },
        path,
    )


def _restore_state(run_root: Path, policy, optimizer, scheduler) -> int:
    path = run_root / "checkpoints" / "last.pt"
    if not path.exists():
        return 0

    state = torch.load(path, map_location="cpu", weights_only=False)
    parameters = dict(policy.named_parameters())
    for name, value in state["trainable_state"].items():
        parameters[name].data.copy_(
            value.to(parameters[name].device, dtype=parameters[name].dtype)
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    print(f"Resumed {run_root} from step {state['step']}")
    return int(state["step"])


def _save_peft(output: Path, policy, preprocessor, postprocessor) -> Path:
    if output.exists():
        return output

    tmp = output.parent / ".final_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    policy.save_pretrained(tmp)
    policy.config.save_pretrained(tmp)
    preprocessor.save_pretrained(tmp)
    postprocessor.save_pretrained(tmp)
    tmp.replace(output)
    shutil.rmtree(output.parent / "checkpoints", ignore_errors=True)
    return output


def _next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_h1_prior(
    seen_checkpoint: str | Path,
    *,
    steps: int,
    batch_size: int,
    seed: int = 7,
    horizon: int = 10,
) -> Path:
    from lerobot.configs.policies import PreTrainedConfig
    from transformers import set_seed

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    run_root = OUTPUTS / "heads" / "h1_dynamics" / f"seed_{seed}"
    final = run_root / "final.pt"
    if final.exists():
        return final

    set_seed(seed)
    policy_config = PreTrainedConfig.from_pretrained(seen_checkpoint)
    dataset = _temporal_dataset(
        SEEN_REPO_ID,
        SEEN_TRAIN_ROOT,
        policy_config,
        horizon,
    )
    policy, preprocessor, _ = _load_policy(
        seen_checkpoint,
        dataset,
        lora=False,
    )
    policy.cuda().eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    checkpoint = run_root / "checkpoints" / "last.pt"
    state = (
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.exists()
        else None
    )
    start_step = int(state["step"]) if state else 0

    loader = _dataloader(dataset, batch_size, seed, start_step)
    iterator = iter(loader)
    latent_dim = int(
        unwrap_policy(policy).model.vlm_with_expert.config.text_config.hidden_size
    )
    config = DynamicsConfig(latent_dim=latent_dim, horizon=horizon)
    head = LatentDynamicsHead(config).cuda()
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    if state:
        head.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])

    log = OUTPUTS / "logs" / "h1_dynamics_prior.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    for step in range(start_step + 1, steps + 1):
        raw, iterator = _next(iterator, loader)
        current = preprocessor(_observation_at(raw, 0))
        future = preprocessor(_observation_at(raw, 1))

        with torch.no_grad():
            z_t = prefix_latent(policy, current).detach()
            z_future = prefix_latent(policy, future).detach()

        actions = raw["action"][:, :horizon].to(
            z_t.device,
            non_blocking=True,
        )
        valid = _future_valid(raw, z_t.device)
        loss = dynamics_loss(head(z_t, actions), z_future, valid=valid)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            message = f"H1 prior step={step} loss={float(loss):.5f}"
            print(message)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

        if step % 500 == 0 and step < steps:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint.with_suffix(".tmp")
            torch.save(
                {
                    "step": step,
                    "state_dict": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                tmp,
            )
            tmp.replace(checkpoint)

    run_root.mkdir(parents=True, exist_ok=True)
    tmp = run_root / ".final.pt"
    torch.save(
        {
            "state_dict": head.state_dict(),
            "config": asdict(config),
            "seen_checkpoint": str(seen_checkpoint),
            "seed": seed,
            "steps": steps,
            "batch_size": batch_size,
        },
        tmp,
    )
    tmp.replace(final)
    shutil.rmtree(run_root / "checkpoints", ignore_errors=True)

    del policy, preprocessor, head, loader, iterator
    gc.collect()
    torch.cuda.empty_cache()
    return final


def train_h1(
    seen_checkpoint: str | Path,
    prior_path: str | Path,
    task_id: int,
    k: int,
    seed: int,
    *,
    steps: int,
    batch_size: int,
    lambda_dyn: float = 0.1,
) -> Path:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.processor.normalize_processor import UnnormalizerProcessorStep
    from transformers import set_seed

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    run_root = (
        OUTPUTS
        / "target"
        / "h1_dynamics"
        / f"t{task_id}"
        / f"k{k}"
        / f"seed_{seed}"
    )
    final = run_root / "final"
    if final.exists():
        return final

    set_seed(seed)
    saved = torch.load(prior_path, map_location="cpu", weights_only=False)
    dynamics_config = DynamicsConfig(**saved["config"])
    policy_config = PreTrainedConfig.from_pretrained(seen_checkpoint)
    dataset = _temporal_dataset(
        target_repo_id(task_id, k),
        subset_root(task_id, k),
        policy_config,
        dynamics_config.horizon,
    )

    policy, preprocessor, postprocessor = _load_policy(
        seen_checkpoint,
        dataset,
        lora=True,
    )
    policy.cuda().train()
    seen_preprocessor = _seen_preprocessor(seen_checkpoint, policy)
    action_unnormalizer = UnnormalizerProcessorStep(
        features=policy.config.output_features,
        norm_map=policy.config.normalization_mapping,
        stats=dataset.meta.stats,
        device="cuda",
    )

    head = LatentDynamicsHead(dynamics_config).cuda()
    head.load_state_dict(saved["state_dict"])
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)

    trainable, optimizer, scheduler, grad_clip = _policy_optimizer(policy, steps)
    start_step = _restore_state(run_root, policy, optimizer, scheduler)
    loader = _dataloader(dataset, batch_size, seed, start_step)
    iterator = iter(loader)

    log = (
        OUTPUTS
        / "logs"
        / f"h1_dynamics__t{task_id}__k{k}__seed_{seed}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)

    for step in range(start_step + 1, steps + 1):
        raw, iterator = _next(iterator, loader)
        current_raw = _observation_at(raw, 0)
        future_raw = _observation_at(raw, 1)

        current = preprocessor(current_raw)
        base = unwrap_policy(policy)
        target = base.prepare_action(current)
        noise = base.model.sample_noise(target.shape, target.device)
        time = base.model.sample_time(len(target), target.device)

        reference_current = seen_preprocessor(current_raw)
        reference_future = seen_preprocessor(future_raw)
        with policy.disable_adapter(), torch.no_grad():
            z_t = prefix_latent(policy, reference_current).detach()
            z_future = prefix_latent(policy, reference_future).detach()

        valid = _future_valid(raw, z_t.device)
        flow, clean_action_normalized = flow_clean_action(
            policy,
            current,
            noise,
            time,
        )
        clean_action_physical = action_unnormalizer(
            {"action": clean_action_normalized}
        )["action"]
        dynamics = dynamics_loss(
            head(
                z_t,
                clean_action_physical[:, :dynamics_config.horizon],
            ),
            z_future,
            valid=valid,
        )
        loss = flow.mean() + lambda_dyn * dynamics

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            message = (
                f"H1 step={step} total={float(loss):.5f} "
                f"flow={float(flow.mean()):.5f} dyn={float(dynamics):.5f}"
            )
            print(message)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

        if step % 500 == 0:
            _save_state(run_root, step, policy, optimizer, scheduler)

    output = _save_peft(
        run_root / "final",
        policy,
        preprocessor,
        postprocessor,
    )
    del policy, preprocessor, postprocessor, head, loader, iterator
    gc.collect()
    torch.cuda.empty_cache()
    return output


def train_h1_grid(
    seen_checkpoint: str | Path,
    prior_path: str | Path,
    steps_by_cell: dict[tuple[int, int], int],
    *,
    batch_size: int,
    lambda_dyn: float = 0.1,
) -> dict[tuple[int, int, int], Path]:
    checkpoints = {}
    for seed in TARGET_SEEDS:
        for task_id in TARGETS:
            for k in BUDGETS:
                checkpoints[(task_id, k, seed)] = train_h1(
                    seen_checkpoint,
                    prior_path,
                    task_id,
                    k,
                    seed,
                    steps=steps_by_cell[(task_id, k)],
                    batch_size=batch_size,
                    lambda_dyn=lambda_dyn,
                )
    return checkpoints
