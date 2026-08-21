from __future__ import annotations

import math
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .h2_progress import setup_timerewarder
from .settings import EXTERNAL, OUTPUTS, TARGETS
from .utils import run


def make_video_pool(
    raw: pd.DataFrame,
    *,
    k: int = 5,
    train_seed: int = 42,
    max_videos_per_checkpoint: int = 10,
) -> pd.DataFrame:
    pool = raw[
        raw["video"].notna()
        & raw["budget"].eq(k)
        & raw["train_seed"].eq(train_seed)
        & raw["method"].isin(["native", "LoRA", "H1", "H2"])
    ].copy()

    pool["checkpoint_id"] = (
        pool["method"].astype(str)
        + "_t"
        + pool["task_id"].astype(str)
        + "_k"
        + pool["budget"].astype(str)
        + "_s"
        + pool["train_seed"].astype(int).astype(str)
    )
    return (
        pool.sort_values(["checkpoint_id", "episode"])
        .groupby("checkpoint_id", group_keys=False)
        .head(max_videos_per_checkpoint)
        .reset_index(drop=True)
    )


def score_timerewarder(
    pool: pd.DataFrame,
    reward_models: dict[tuple[int, int], Path],
    *,
    k: int = 5,
) -> pd.DataFrame:
    repo = setup_timerewarder()
    python = repo / ".venv" / "bin" / "python"
    runner = Path(__file__).with_name("timerewarder_runner.py").resolve()
    work = OUTPUTS / "bonus_b" / "timerewarder"
    work.mkdir(parents=True, exist_ok=True)
    parts = []

    for task_id in TARGETS:
        task_pool = pool[pool["task_id"].eq(task_id)].copy()
        if task_pool.empty:
            continue

        output = work / f"t{task_id}_k{k}.csv"
        if output.exists():
            scored = pd.read_csv(output)
            parts.append(scored)
            continue

        manifest = work / f"t{task_id}_k{k}_manifest.csv"
        scores = work / f"t{task_id}_k{k}_scores.csv"
        frame = task_pool.reset_index(drop=True)
        frame["row_id"] = np.arange(len(frame))
        frame[["row_id", "video"]].to_csv(manifest, index=False)

        run(
            [
                str(python),
                str(runner),
                "pool",
                "--repo",
                str(repo.resolve()),
                "--checkpoint",
                str(reward_models[(task_id, k)].resolve()),
                "--manifest",
                str(manifest.resolve()),
                "--out",
                str(scores.resolve()),
            ],
            log_path=OUTPUTS
            / "logs"
            / f"bonus_timerewarder__t{task_id}__k{k}.log",
        )

        reward = pd.read_csv(scores)
        scored = (
            frame.merge(reward, on="row_id", validate="one_to_one")
            .drop(columns="row_id")
        )
        scored["critic"] = "TimeRewarder"
        scored.to_csv(output, index=False)
        manifest.unlink(missing_ok=True)
        scores.unlink(missing_ok=True)
        parts.append(scored)

    return pd.concat(parts, ignore_index=True)


def setup_robometer() -> Path:
    repo = EXTERNAL / "robometer"
    if not repo.exists():
        EXTERNAL.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "https://github.com/robometer/robometer.git",
                str(repo),
            ]
        )

    run(
        ["git", "log", "-1", "--format=Robometer:%H"],
        cwd=repo,
        log_path=OUTPUTS / "logs" / "external_versions.log",
    )

    marker = repo / ".robometer_env_ready"
    if marker.exists():
        return repo

    if shutil.which("uv") is None:
        run([sys.executable, "-m", "pip", "install", "-q", "uv"])
    run(["uv", "python", "install", "3.10"], cwd=repo)
    run(["uv", "sync"], cwd=repo)
    run(
        [
            "uv",
            "run",
            "python",
            "-c",
            (
                "import torch, robometer; "
                "assert torch.cuda.is_available(); "
                "assert torch.version.cuda == '12.8'; "
                "print(torch.__version__, torch.version.cuda)"
            ),
        ],
        cwd=repo,
    )
    marker.write_text("ready\n", encoding="utf-8")
    return repo


def start_robometer(
    *,
    model: str = "aliangdw/Robometer-4B-LIBERO",
    port: int = 8000,
    startup_timeout_s: int = 300,
):
    repo = setup_robometer()
    log_path = OUTPUTS / "logs" / "robometer_server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8")

    process = subprocess.Popen(
        [
            "uv",
            "run",
            "python",
            "robometer/evals/eval_server.py",
            f"model_path={model}",
            "num_gpus=1",
            f"server_port={port}",
        ],
        cwd=repo,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + startup_timeout_s

    while time.time() < deadline:
        if process.poll() is not None:
            log.close()
            raise RuntimeError(
                f"Robometer server exited early; inspect {log_path}"
            )
        try:
            with urllib.request.urlopen(
                url + "/health",
                timeout=2,
            ) as response:
                if response.status == 200:
                    return process, url, log
        except Exception:
            time.sleep(2)

    process.terminate()
    process.wait(timeout=30)
    log.close()
    raise TimeoutError(f"Robometer server did not start; inspect {log_path}")


def stop_robometer(process, log) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    if log is not None and not log.closed:
        log.close()


def _robometer_score(progress_path: Path) -> float:
    progress = np.asarray(np.load(progress_path), dtype=float).reshape(-1)
    if len(progress) == 0:
        return float("nan")
    start = max(0, int(math.floor(0.8 * len(progress))))
    return float(np.mean(progress[start:]))


def score_robometer(
    pool: pd.DataFrame,
    *,
    model: str = "aliangdw/Robometer-4B-LIBERO",
    port: int = 8000,
) -> pd.DataFrame:
    repo = setup_robometer()
    output_dir = OUTPUTS / "bonus_b" / "robometer"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    process = log = None

    try:
        process, server_url, log = start_robometer(
            model=model,
            port=port,
        )
        for row in pool.itertuples(index=False):
            output = output_dir / (
                f"{row.checkpoint_id}_ep{int(row.episode):03d}.npy"
            )
            if not output.exists():
                run(
                    [
                        "uv",
                        "run",
                        "python",
                        "scripts/example_inference.py",
                        "--eval-server-url",
                        server_url,
                        "--video",
                        str(Path(row.video).resolve()),
                        "--task",
                        TARGETS[int(row.task_id)],
                        "--fps",
                        "1.0",
                        "--out",
                        str(output.resolve()),
                    ],
                    cwd=repo,
                    log_path=OUTPUTS
                    / "logs"
                    / "bonus_robometer_client.log",
                )

            record = row._asdict()
            record["reward_score"] = _robometer_score(output)
            record["critic"] = "Robometer-4B-LIBERO"
            rows.append(record)
    finally:
        stop_robometer(process, log)

    return pd.DataFrame(rows)


def checkpoint_success(
    raw: pd.DataFrame,
    *,
    k: int = 5,
    train_seed: int = 42,
) -> pd.DataFrame:
    frame = raw[
        raw["budget"].eq(k)
        & raw["train_seed"].eq(train_seed)
        & raw["method"].isin(["native", "LoRA", "H1", "H2"])
    ].copy()
    frame["checkpoint_id"] = (
        frame["method"].astype(str)
        + "_t"
        + frame["task_id"].astype(str)
        + "_k"
        + frame["budget"].astype(str)
        + "_s"
        + frame["train_seed"].astype(int).astype(str)
    )
    return (
        frame.groupby(
            [
                "checkpoint_id",
                "method",
                "task_id",
                "budget",
                "train_seed",
            ],
            as_index=False,
        )
        .agg(true_success=("success", "mean"))
    )


def rank_checkpoints(
    scores: pd.DataFrame,
    success: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reward = (
        scores.groupby(
            [
                "critic",
                "checkpoint_id",
                "method",
                "task_id",
                "budget",
            ],
            as_index=False,
        )
        .agg(reward_score=("reward_score", "mean"))
    )
    checkpoints = reward.merge(
        success[
            [
                "checkpoint_id",
                "true_success",
            ]
        ],
        on="checkpoint_id",
        how="left",
        validate="many_to_one",
    )

    rows = []
    for (critic, task_id, budget), group in checkpoints.groupby(
        ["critic", "task_id", "budget"]
    ):
        if len(group) < 3:
            continue

        spearman = spearmanr(
            group["reward_score"],
            group["true_success"],
        ).statistic
        kendall = kendalltau(
            group["reward_score"],
            group["true_success"],
        ).statistic

        comparable = 0
        correct = 0
        values = group.reset_index(drop=True)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                success_delta = (
                    values.loc[i, "true_success"]
                    - values.loc[j, "true_success"]
                )
                if success_delta == 0:
                    continue
                reward_delta = (
                    values.loc[i, "reward_score"]
                    - values.loc[j, "reward_score"]
                )
                comparable += 1
                correct += int(
                    np.sign(success_delta) == np.sign(reward_delta)
                )

        rows.append(
            {
                "critic": critic,
                "task_id": task_id,
                "budget": budget,
                "spearman": float(spearman),
                "kendall": float(kendall),
                "pairwise_accuracy": (
                    correct / comparable
                    if comparable
                    else np.nan
                ),
                "checkpoints": len(group),
            }
        )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return checkpoints, ranking

    macro = (
        ranking.groupby("critic")[
            ["spearman", "kendall", "pairwise_accuracy"]
        ]
        .mean()
        .reset_index()
    )
    macro["task_id"] = "macro"
    macro["budget"] = "macro"
    macro["checkpoints"] = np.nan
    ranking = pd.concat([ranking, macro], ignore_index=True)
    return checkpoints, ranking


def reward_pressure(
    checkpoint_scores: pd.DataFrame,
    *,
    critic: str = "TimeRewarder",
) -> pd.DataFrame:
    frame = checkpoint_scores[
        checkpoint_scores["critic"].eq(critic)
        & checkpoint_scores["method"].isin(["LoRA", "H2"])
    ].copy()

    rows = []
    for task_id, group in frame.groupby("task_id"):
        by_method = group.set_index("method")
        if not {"LoRA", "H2"}.issubset(by_method.index):
            continue
        rows.append(
            {
                "task_id": task_id,
                "lora_reward": by_method.loc["LoRA", "reward_score"],
                "h2_reward": by_method.loc["H2", "reward_score"],
                "delta_reward": (
                    by_method.loc["H2", "reward_score"]
                    - by_method.loc["LoRA", "reward_score"]
                ),
                "lora_success": by_method.loc["LoRA", "true_success"],
                "h2_success": by_method.loc["H2", "true_success"],
                "delta_success": (
                    by_method.loc["H2", "true_success"]
                    - by_method.loc["LoRA", "true_success"]
                ),
            }
        )
    return pd.DataFrame(rows)
