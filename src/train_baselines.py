from __future__ import annotations

import json
import shutil
from pathlib import Path

from .data import target_training_source
from .settings import (
    BASE_MODEL,
    BUDGETS,
    DATASET_RENAME_MAP,
    OUTPUTS,
    SEEN_REPO_ID,
    SEEN_TRAIN_ROOT,
    TARGETS,
    TARGET_SEEDS,
    VIDEO_BACKEND,
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
    episodes: list[int] | None = None,
) -> Path:
    """Run the stock LeRobot/SmolVLA fine-tune without adapter wrapping."""
    final = output_dir / "final"
    if final.exists():
        return final

    log = _log_path(output_dir)
    last = output_dir / "checkpoints" / "last" / "pretrained_model"

    if output_dir.exists():
        config = last / "train_config.json"
        if not config.exists():
            raise FileExistsError(
                f"Cannot safely resume {output_dir}: {config} is missing. "
                "Remove this partial run directory and start this cell again."
            )
        run(
            [
                "lerobot-train",
                f"--config_path={config}",
                "--resume=true",
            ],
            log_path=log,
        )
    else:
        command = [
            "lerobot-train",
            f"--policy.path={policy_path}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={dataset_root}",
            f"--dataset.video_backend={VIDEO_BACKEND}",
            f"--output_dir={output_dir}",
            "--policy.device=cuda",
            "--policy.push_to_hub=false",
            "--wandb.enable=false",
            "--policy.empty_cameras=1",
            f"--rename_map={json.dumps(DATASET_RENAME_MAP)}",
            f"--steps={int(steps)}",
            f"--batch_size={int(batch_size)}",
            f"--policy.scheduler_decay_steps={int(steps)}",
            f"--policy.scheduler_warmup_steps={min(1000, max(100, int(steps) // 10))}",
            "--save_checkpoint=true",
            f"--save_freq={max(250, int(steps) // 2)}",
            f"--seed={int(seed)}",
        ]
        if episodes is not None:
            command.append(f"--dataset.episodes={json.dumps([int(x) for x in episodes])}")
        run(command, log_path=log)

    last = output_dir / "checkpoints" / "last" / "pretrained_model"
    if not last.exists():
        raise FileNotFoundError(
            f"LeRobot finished without {last}; inspect {log}."
        )

    tmp = output_dir / ".final_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(last.resolve(), tmp)
    tmp.replace(final)
    # Only the deployable final checkpoint is retained after a successful run.
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
    )


def train_target_baseline(
    seen_checkpoint: str | Path,
    task_id: int,
    k: int,
    seed: int,
    *,
    steps: int,
    batch_size: int,
) -> Path:
    repo_id, root, episodes = target_training_source(task_id, k)
    return train_cli(
        seen_checkpoint,
        root,
        repo_id,
        OUTPUTS / "target" / "baseline" / f"t{task_id}" / f"k{k}" / f"seed_{seed}",
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        episodes=episodes,
    )


def train_baseline_grid(
    seen_checkpoint: str | Path,
    steps_by_cell: dict[tuple[int, int], int],
    *,
    batch_size: int,
) -> dict[tuple[int, int, int], Path]:
    checkpoints = {}
    for seed in TARGET_SEEDS:
        for task_id in TARGETS:
            for k in BUDGETS:
                checkpoints[(task_id, k, seed)] = train_target_baseline(
                    seen_checkpoint,
                    task_id,
                    k,
                    seed,
                    steps=steps_by_cell[(task_id, k)],
                    batch_size=batch_size,
                )
    return checkpoints
