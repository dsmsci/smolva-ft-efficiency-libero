from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import subset_root, target_repo_id
from .settings import (
    BUDGETS,
    DATA,
    EXTERNAL,
    OUTPUTS,
    TARGETS,
    TARGET_SEEDS,
)
from .train_baselines import train_cli
from .utils import run


def prepare_timerewarder_videos(
    task_id: int,
    k: int,
    camera_key: str = "observation.images.image",
) -> Path:
    import imageio.v2 as imageio
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        target_repo_id(task_id, k),
        root=subset_root(task_id, k),
    )
    task_root = DATA / "timerewarder" / f"t{task_id}_k{k}"
    videos = task_root / "videos"
    videos.mkdir(parents=True, exist_ok=True)

    names = []
    for episode in range(dataset.meta.total_episodes):
        name = f"episode_{episode:03d}.mp4"
        names.append(name)
        destination = videos / name
        if destination.exists():
            continue

        start = int(dataset.meta.episodes[episode]["dataset_from_index"])
        stop = int(dataset.meta.episodes[episode]["dataset_to_index"])
        with imageio.get_writer(
            destination,
            fps=dataset.meta.fps,
            codec="libx264",
        ) as writer:
            for index in range(start, stop):
                image = dataset[index][camera_key]
                if hasattr(image, "detach"):
                    image = image.detach().cpu().numpy()
                if image.ndim == 3 and image.shape[0] in (1, 3):
                    image = np.moveaxis(image, 0, -1)
                if image.dtype != np.uint8:
                    if float(np.nanmax(image)) <= 1.0:
                        image = image * 255.0
                    image = np.clip(image, 0, 255).astype(np.uint8)
                writer.append_data(image)

    train_count = {5: 4, 10: 8, 25: 20}[k]
    (task_root / "label.txt").write_text(
        "\n".join(names[:train_count]) + "\n",
        encoding="utf-8",
    )
    (task_root / "label_val.txt").write_text(
        "\n".join(names[train_count:]) + "\n",
        encoding="utf-8",
    )
    return task_root


def setup_timerewarder() -> Path:
    repo = EXTERNAL / "TimeRewarder"
    if not repo.exists():
        EXTERNAL.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "https://github.com/CowAndSheep/TimeRewarder.git",
                str(repo),
            ]
        )

    run(
        ["git", "log", "-1", "--format=TimeRewarder:%H"],
        cwd=repo,
        log_path=OUTPUTS / "logs" / "external_versions.log",
    )

    if shutil.which("uv") is None:
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])

    venv = repo / ".venv"
    marker = venv / ".blackwell_py39_torch271_cu128"
    if marker.exists():
        return repo

    shutil.rmtree(venv, ignore_errors=True)
    run(["uv", "python", "install", "3.9"])
    run(["uv", "venv", "--python", "3.9", str(venv)])

    python = venv / "bin" / "python"
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "torch==2.7.1+cu128",
            "torchvision==0.22.1+cu128",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ]
    )
    run(
        [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            "numpy==1.24.4",
            "einops==0.8.0",
            "timm==1.0.9",
            "ftfy==6.2.3",
            "regex==2024.9.11",
            "decord==0.6.0",
            "mmcv==1.7.1",
            "opencv-python==4.5.3.56",
            "yacs==0.1.8",
            "tensorboard==2.14.0",
            "scipy==1.10.1",
            "imageio==2.9.0",
            "imageio-ffmpeg==0.4.4",
            "pandas==1.3.0",
            "matplotlib==3.4.2",
            "termcolor==1.1.0",
            "protobuf==3.20.1",
            "huggingface_hub==0.25.1",
        ]
    )
    run([str(python), "-m", "pip", "check"])
    run(
        [
            str(python),
            "-c",
            (
                "import torch, torchvision, decord, mmcv; "
                "from models import clip_withhead; "
                "assert torch.cuda.is_available(); "
                "assert torch.version.cuda == '12.8'; "
                "print(torch.__version__, torch.version.cuda)"
            ),
        ],
        cwd=repo,
    )
    marker.write_text("ready\n", encoding="utf-8")
    return repo

def _latest_checkpoint(output: Path) -> Path | None:
    candidates = list(output.glob("ckpt_epoch_*.pth"))
    if not candidates:
        return None

    def epoch(path: Path) -> int:
        match = re.search(r"ckpt_epoch_(\d+)", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=epoch)


def train_timerewarder(
    task_id: int,
    k: int,
    *,
    seed: int = 7,
    epochs: int = 20,
    batch_size: int = 16,
) -> Path:
    task_root = prepare_timerewarder_videos(task_id, k)
    repo = setup_timerewarder()
    python = repo / ".venv" / "bin" / "python"
    output = (
        OUTPUTS
        / "reward"
        / "timerewarder"
        / f"t{task_id}_k{k}_seed_{seed}"
    )
    output.mkdir(parents=True, exist_ok=True)

    expected = output / f"ckpt_epoch_{epochs - 1}.pth"
    if expected.exists():
        return expected

    log = OUTPUTS / "logs" / f"timerewarder__t{task_id}__k{k}.log"
    run(
        [
            str(python),
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "main.py",
            "-cfg",
            "configs/metaworld.yaml",
            "--output",
            str(output.resolve()),
            "--opts",
            "DATA.ROOT",
            str((task_root / "videos").resolve()),
            "DATA.TRAIN_FILE",
            str((task_root / "label.txt").resolve()),
            "DATA.VAL_FILE",
            str((task_root / "label_val.txt").resolve()),
            "MODEL.BIN_NUM",
            "20",
            "MODEL.USE_TEXT",
            "False",
            "MODEL.DIFF_REPRESENTATION",
            "False",
            "MODEL.IMPLICIT_NEGATIVE",
            "True",
            "DATA.WEIGHTED_SAMPLE",
            "True",
            "SEED",
            str(seed),
            "SAVE_FREQ",
            "100",
            "TRAIN.EPOCHS",
            str(epochs),
            "TRAIN.BATCH_SIZE",
            str(batch_size),
        ],
        cwd=repo / "training",
        log_path=log,
    )

    if not expected.exists():
        checkpoint = _latest_checkpoint(output)
        raise FileNotFoundError(
            f"Expected final TimeRewarder checkpoint {expected}; "
            f"latest found: {checkpoint}"
        )
    return expected

def export_progress(
    task_id: int,
    k: int,
    checkpoint: str | Path,
    *,
    seed: int = 7,
) -> Path:
    repo = setup_timerewarder()
    python = repo / ".venv" / "bin" / "python"
    task_root = DATA / "timerewarder" / f"t{task_id}_k{k}"
    if not task_root.exists():
        prepare_timerewarder_videos(task_id, k)

    output = (
        OUTPUTS
        / "reward"
        / "progress"
        / f"t{task_id}_k{k}_seed_{seed}.parquet"
    )
    if output.exists():
        return output

    output.parent.mkdir(parents=True, exist_ok=True)
    raw_csv = output.with_suffix(".csv")
    runner = Path(__file__).with_name("timerewarder_runner.py").resolve()
    run(
        [
            str(python),
            str(runner),
            "progress",
            "--repo",
            str(repo.resolve()),
            "--checkpoint",
            str(Path(checkpoint).resolve()),
            "--videos",
            str((task_root / "videos").resolve()),
            "--out",
            str(raw_csv.resolve()),
        ],
        log_path=OUTPUTS
        / "logs"
        / f"timerewarder_progress__t{task_id}__k{k}.log",
    )

    root = subset_root(task_id, k)
    parts = sorted((root / "data").rglob("*.parquet"))
    index_table = pd.concat(
        [
            pd.read_parquet(
                part,
                columns=["index", "episode_index", "frame_index"],
            )
            for part in parts
        ],
        ignore_index=True,
    )
    reward_table = pd.read_csv(raw_csv)
    progress = index_table.merge(
        reward_table,
        on=["episode_index", "frame_index"],
        how="inner",
        validate="one_to_one",
    )
    if len(progress) != len(index_table):
        raise RuntimeError(
            f"TimeRewarder progress covers {len(progress)} of "
            f"{len(index_table)} target frames"
        )
    if (
        not progress["index"].is_unique
        or not progress["progress_sparse"].between(0, 1).all()
    ):
        raise RuntimeError("Invalid TimeRewarder progress table")

    progress.sort_values("index").to_parquet(output, index=False)
    raw_csv.unlink(missing_ok=True)
    return output


def prepare_h2_rewards(
    *,
    seed: int = 7,
    epochs: int = 20,
    batch_size: int = 16,
) -> tuple[dict[tuple[int, int], Path], dict[tuple[int, int], Path]]:
    reward_models = {}
    progress_files = {}
    for task_id in TARGETS:
        for k in BUDGETS:
            checkpoint = train_timerewarder(
                task_id,
                k,
                seed=seed,
                epochs=epochs,
                batch_size=batch_size,
            )
            progress = export_progress(
                task_id,
                k,
                checkpoint,
                seed=seed,
            )
            reward_models[(task_id, k)] = checkpoint
            progress_files[(task_id, k)] = progress
    return reward_models, progress_files


def train_h2(
    seen_checkpoint: str | Path,
    task_id: int,
    k: int,
    seed: int,
    progress_path: str | Path,
    *,
    steps: int,
    batch_size: int,
    kappa: float = 0.01,
) -> Path:
    output = (
        OUTPUTS
        / "target"
        / "h2_progress"
        / f"t{task_id}"
        / f"k{k}"
        / f"seed_{seed}"
    )
    return train_cli(
        seen_checkpoint,
        subset_root(task_id, k),
        target_repo_id(task_id, k),
        output,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        lora=True,
        sample_weighting={
            "type": "rabc",
            "head_mode": "sparse",
            "kappa": kappa,
            "progress_path": Path(progress_path).resolve(),
        },
        save_freq=max(100, min(500, steps // 3)),
    )


def train_h2_grid(
    seen_checkpoint: str | Path,
    progress_files: dict[tuple[int, int], Path],
    steps_by_cell: dict[tuple[int, int], int],
    *,
    batch_size: int,
    kappa: float = 0.01,
) -> dict[tuple[int, int, int], Path]:
    checkpoints = {}
    for seed in TARGET_SEEDS:
        for task_id in TARGETS:
            for k in BUDGETS:
                checkpoints[(task_id, k, seed)] = train_h2(
                    seen_checkpoint,
                    task_id,
                    k,
                    seed,
                    progress_files[(task_id, k)],
                    steps=steps_by_cell[(task_id, k)],
                    batch_size=batch_size,
                    kappa=kappa,
                )
    return checkpoints
