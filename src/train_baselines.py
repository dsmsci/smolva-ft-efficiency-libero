from __future__ import annotations

import shutil
from pathlib import Path

from .data import subset_root, target_repo_id
from .settings import (
    BASE_MODEL,
    BUDGETS,
    OUTPUTS,
    SEEN_REPO_ID,
    SEEN_TRAIN_ROOT,
    TARGETS,
    TARGET_SEEDS,
)
from .utils import run


def _log_path(output_dir: Path) -> Path:
    relative = output_dir.relative_to(OUTPUTS).as_posix().replace("/", "__")
    return OUTPUTS / "logs" / f"{relative}.log"


def train_cli(
    policy_path: str | Path,
    dataset_root: Path,
    repo_id: str,
    output_dir: Path,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    lora: bool = False,
    sample_weighting: dict | None = None,
    save_freq: int = 1000,
) -> Path:
    final = output_dir / "final"
    if final.exists():
        return final

    log = _log_path(output_dir)
    last = output_dir / "checkpoints" / "last" / "pretrained_model"

    if output_dir.exists():
        config = last / "train_config.json"
        if not config.exists():
            raise FileExistsError(
                f"Cannot resume {output_dir}; remove the partial directory."
            )
        run(
            [
                "lerobot-train",
                "--resume=true",
                f"--config_path={config}",
                f"--output_dir={output_dir}",
            ],
            log_path=log,
        )
    else:
        command = [
            "lerobot-train",
            f"--policy.path={policy_path}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={dataset_root}",
            f"--output_dir={output_dir}",
            "--policy.device=cuda",
            "--policy.push_to_hub=false",
            "--wandb.enable=false",
            f"--steps={steps}",
            f"--batch_size={batch_size}",
            f"--policy.scheduler_decay_steps={steps}",
            f"--policy.scheduler_warmup_steps={min(1000, max(100, steps // 10))}",
            "--save_checkpoint=true",
            f"--save_freq={save_freq}",
            f"--seed={seed}",
        ]
        if lora:
            command += [
                "--peft.method_type=LORA",
                "--peft.r=64",
                "--peft.lora_alpha=64",
            ]
        if sample_weighting:
            command += [
                f"--sample_weighting.{key}={value}"
                for key, value in sample_weighting.items()
            ]
        run(command, log_path=log)

    last = output_dir / "checkpoints" / "last" / "pretrained_model"
    if not last.exists():
        raise FileNotFoundError(last)

    tmp = output_dir / ".final_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(last.resolve(), tmp)
    tmp.replace(final)
    shutil.rmtree(output_dir / "checkpoints", ignore_errors=True)
    return final


def train_seen(*, steps: int, batch_size: int, seed: int = 7) -> Path:
    return train_cli(
        BASE_MODEL,
        SEEN_TRAIN_ROOT,
        SEEN_REPO_ID,
        OUTPUTS / "seen" / f"seed_{seed}",
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        save_freq=max(250, min(5000, steps // 4)),
    )


def train_target_native(
    seen_checkpoint: str | Path,
    task_id: int,
    k: int,
    seed: int,
    *,
    steps: int,
    batch_size: int,
) -> Path:
    return train_cli(
        seen_checkpoint,
        subset_root(task_id, k),
        target_repo_id(task_id, k),
        OUTPUTS / "target" / "native" / f"t{task_id}" / f"k{k}" / f"seed_{seed}",
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        save_freq=max(100, min(500, steps // 3)),
    )


def train_target_lora(
    seen_checkpoint: str | Path,
    task_id: int,
    k: int,
    seed: int,
    *,
    steps: int,
    batch_size: int,
) -> Path:
    return train_cli(
        seen_checkpoint,
        subset_root(task_id, k),
        target_repo_id(task_id, k),
        OUTPUTS / "target" / "lora" / f"t{task_id}" / f"k{k}" / f"seed_{seed}",
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        lora=True,
        save_freq=max(100, min(500, steps // 3)),
    )


def train_native_grid(
    seen_checkpoint: str | Path,
    steps_by_cell: dict[tuple[int, int], int],
    *,
    batch_size: int,
) -> dict[tuple[int, int, int], Path]:
    checkpoints = {}
    for seed in TARGET_SEEDS:
        for task_id in TARGETS:
            for k in BUDGETS:
                checkpoints[(task_id, k, seed)] = train_target_native(
                    seen_checkpoint,
                    task_id,
                    k,
                    seed,
                    steps=steps_by_cell[(task_id, k)],
                    batch_size=batch_size,
                )
    return checkpoints


def train_lora_grid(
    seen_checkpoint: str | Path,
    steps_by_cell: dict[tuple[int, int], int],
    *,
    batch_size: int,
) -> dict[tuple[int, int, int], Path]:
    checkpoints = {}
    for seed in TARGET_SEEDS:
        for task_id in TARGETS:
            for k in BUDGETS:
                checkpoints[(task_id, k, seed)] = train_target_lora(
                    seen_checkpoint,
                    task_id,
                    k,
                    seed,
                    steps=steps_by_cell[(task_id, k)],
                    batch_size=batch_size,
                )
    return checkpoints
