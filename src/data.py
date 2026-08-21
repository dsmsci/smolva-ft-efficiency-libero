from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .settings import (
    BUDGETS,
    DATA,
    DATASET_ID,
    FIXED,
    RAW,
    SEEN_REPO_ID,
    SEEN_TRAIN_ROOT,
    SUBSETS,
    TARGETS,
)

H264 = DATA / "h264"


def _copy_dataset_for_mutation(src: Path, dst: Path) -> None:
    if dst.exists():
        return

    def copy_file(source, destination):
        if Path(source).suffix.lower() == ".mp4":
            try:
                os.link(source, destination)
                return destination
            except OSError:
                pass
        return shutil.copy2(source, destination)

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp, copy_function=copy_file)
    tmp.replace(dst)


def _download_suite(suite: str) -> Path:
    from huggingface_hub import snapshot_download

    repo_dir = Path(
        snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            allow_patterns=[f"{suite}/**"],
            local_dir=RAW,
        )
    )
    root = repo_dir / suite
    if not (root / "meta" / "info.json").exists():
        raise FileNotFoundError(root)
    return root


def _video_codec(path: Path) -> str:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _copy_non_video_files(src: Path, dst: Path) -> None:
    def ignore_videos(directory, names):
        return [name for name in names if name.lower().endswith(".mp4")]

    shutil.copytree(
        src,
        dst,
        dirs_exist_ok=True,
        ignore=ignore_videos,
    )


def _patch_h264_metadata(root: Path) -> None:
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(info_path)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    info.pop("data_files", None)

    video_features = 0
    for feature in info.get("features", {}).values():
        if feature.get("dtype") != "video":
            continue

        video_features += 1
        video = feature.setdefault("info", {})
        video["video.codec"] = "h264"
        video["video.pix_fmt"] = "yuv420p"
        video["video.g"] = 2
        video["video.crf"] = 18
        video["video.fast_decode"] = 0
        video.pop("video.preset", None)
        video.pop("video.extra_options", None)

    if video_features == 0:
        raise RuntimeError(f"No video features in {info_path}")

    info_path.write_text(
        json.dumps(info, indent=4) + "\n",
        encoding="utf-8",
    )


def _h264_videos_are_complete(src: Path, dst: Path) -> bool:
    source_videos = sorted(
        path.relative_to(src)
        for path in src.rglob("*.mp4")
    )
    target_videos = sorted(
        path.relative_to(dst)
        for path in dst.rglob("*.mp4")
    )

    if not source_videos or target_videos != source_videos:
        return False

    try:
        return all(
            _video_codec(dst / relative) == "h264"
            for relative in target_videos
        )
    except (OSError, subprocess.CalledProcessError):
        return False


def _prepare_h264_suite(src: Path, dst: Path) -> Path:
    marker = dst / ".h264_source_ready"

    if marker.exists():
        if not (dst / "meta" / "info.json").exists():
            raise FileNotFoundError(dst / "meta" / "info.json")
        return dst

    # Reuse a complete manual H264 transcode if it already exists.
    if dst.exists() and _h264_videos_are_complete(src, dst):
        _copy_non_video_files(src, dst)
        _patch_h264_metadata(dst)
        marker.write_text("av1_cuvid -> h264_nvenc\n", encoding="utf-8")
        print(f"H264 source reused: {dst}")
        return dst

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(dst, ignore_errors=True)
    _copy_non_video_files(src, tmp)

    source_videos = sorted(src.rglob("*.mp4"))
    if not source_videos:
        raise RuntimeError(f"No MP4 videos found in {src}")

    for index, source_video in enumerate(source_videos, start=1):
        relative = source_video.relative_to(src)
        target_video = tmp / relative
        target_video.parent.mkdir(parents=True, exist_ok=True)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-c:v",
                "av1_cuvid",
                "-i",
                str(source_video),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "h264_nvenc",
                "-preset",
                "p4",
                "-cq",
                "18",
                "-g",
                "2",
                "-bf",
                "0",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(target_video),
            ],
            check=True,
        )

        if _video_codec(target_video) != "h264":
            raise RuntimeError(
                f"Transcode did not produce H264: {target_video}"
            )

        print(
            f"[{index}/{len(source_videos)}] AV1 -> H264: "
            f"{source_video}"
        )

    _patch_h264_metadata(tmp)
    (tmp / ".h264_source_ready").write_text(
        "av1_cuvid -> h264_nvenc\n",
        encoding="utf-8",
    )
    tmp.replace(dst)
    print(f"H264 source ready: {len(source_videos)} videos")
    return dst


def _convert_gripper_in_place(root: Path) -> None:
    marker = root / ".gripper_env_convention"
    if marker.exists():
        return

    for parquet in sorted((root / "data").rglob("*.parquet")):
        frame = pd.read_parquet(parquet)
        if "action" not in frame.columns:
            raise KeyError(f"{parquet}: action column missing")

        def convert(action):
            action = np.asarray(action, dtype=np.float32).copy()
            if action.shape[-1] != 7:
                raise ValueError(
                    f"Expected action dim 7, got {action.shape}"
                )
            gripper = np.asarray(action[..., -1])
            if gripper.min() < -1e-4 or gripper.max() > 1.0001:
                raise ValueError(
                    f"Source gripper must be in [0,1], got "
                    f"[{gripper.min()},{gripper.max()}]"
                )
            action[..., -1] = 1.0 - 2.0 * gripper
            return action.tolist()

        frame["action"] = frame["action"].map(convert)
        frame.to_parquet(parquet, index=False)

    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        f"local/{root.name}_fixed",
        root=root,
    )
    recompute_stats(dataset, skip_image_video=True)
    marker.write_text(
        "g_env=1-2*g_data\n",
        encoding="utf-8",
    )


def _episodes_for_instruction(dataset, instruction: str) -> list[int]:
    selected = []
    for episode in range(dataset.meta.total_episodes):
        tasks = [
            str(x).strip()
            for x in dataset.meta.episodes[episode]["tasks"]
        ]
        if instruction in tasks:
            selected.append(episode)
    return selected


def target_repo_id(task_id: int, k: int) -> str:
    return f"local/libero_goal_fixed_t{task_id}_k{k}"


def subset_root(task_id: int, k: int) -> Path:
    if task_id not in TARGETS or k not in BUDGETS:
        raise ValueError((task_id, k))
    root = SUBSETS / f"t{task_id}_k{k}"
    if not root.exists():
        raise FileNotFoundError(f"{root}; run prepare_data()")
    return root


def _subset_is_complete(root: Path, task_id: int, k: int) -> bool:
    if not root.exists():
        return False

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(
            target_repo_id(task_id, k),
            root=root,
        )
        return int(dataset.meta.total_episodes) == k
    except Exception:
        return False


def prepare_data(force: bool = False) -> dict:
    if force:
        shutil.rmtree(FIXED, ignore_errors=True)
        shutil.rmtree(SUBSETS, ignore_errors=True)

    RAW.mkdir(parents=True, exist_ok=True)
    H264.mkdir(parents=True, exist_ok=True)
    FIXED.mkdir(parents=True, exist_ok=True)
    SUBSETS.mkdir(parents=True, exist_ok=True)

    seen_raw = _download_suite("libero_90")
    goal_raw = _download_suite("libero_goal")

    seen_h264 = _prepare_h264_suite(
        seen_raw,
        H264 / "libero_90",
    )
    goal_h264 = _prepare_h264_suite(
        goal_raw,
        H264 / "libero_goal",
    )

    seen = FIXED / "libero_90"
    goal = FIXED / "libero_goal"

    for source, destination in (
        (seen_h264, seen),
        (goal_h264, goal),
    ):
        marker = destination / ".gripper_env_convention"
        if destination.exists() and not marker.exists():
            shutil.rmtree(destination)

        _copy_dataset_for_mutation(source, destination)
        _convert_gripper_in_place(destination)

    from lerobot.datasets.dataset_tools import (
        recompute_stats,
        split_dataset,
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    seen_dataset = LeRobotDataset(
        SEEN_REPO_ID,
        root=seen,
    )
    goal_dataset = LeRobotDataset(
        "local/libero_goal_fixed",
        root=goal,
    )

    for dataset_name, dataset in (
        ("seen", seen_dataset),
        ("goal", goal_dataset),
    ):
        for video_key in dataset.meta.video_keys:
            codec = dataset.meta.info.features[video_key]["info"].get(
                "video.codec"
            )
            if codec != "h264":
                raise RuntimeError(
                    f"{dataset_name} {video_key}: expected h264, got {codec}"
                )

    manifest_rows = []

    for task_id, instruction in TARGETS.items():
        episodes = _episodes_for_instruction(
            goal_dataset,
            instruction,
        )
        if len(episodes) < 25:
            raise RuntimeError(
                f"{instruction}: only {len(episodes)} episodes"
            )

        for k in BUDGETS:
            chosen = episodes[:k]
            split_name = f"t{task_id}_k{k}"
            output = SUBSETS / split_name

            for subset_episode, source_episode in enumerate(chosen):
                manifest_rows.append(
                    {
                        "task_id": task_id,
                        "instruction": instruction,
                        "K": k,
                        "subset_episode": subset_episode,
                        "source_episode": source_episode,
                    }
                )

            if not _subset_is_complete(output, task_id, k):
                shutil.rmtree(output, ignore_errors=True)
                made = split_dataset(
                    goal_dataset,
                    {split_name: chosen},
                    output_dir=SUBSETS,
                )[split_name]

                if made.root != output:
                    raise RuntimeError(
                        f"Unexpected split root: "
                        f"{made.root} != {output}"
                    )

                recompute_stats(
                    made,
                    skip_image_video=True,
                )

    manifest_path = DATA / "subset_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(
        manifest_path,
        index=False,
    )

    target_frames = {}
    for task_id, instruction in TARGETS.items():
        for k in BUDGETS:
            dataset = LeRobotDataset(
                target_repo_id(task_id, k),
                root=subset_root(task_id, k),
            )
            if int(dataset.meta.total_episodes) != k:
                raise RuntimeError(
                    f"t{task_id}_k{k}: "
                    f"{dataset.meta.total_episodes} episodes, "
                    f"expected {k}"
                )

            for episode in range(dataset.meta.total_episodes):
                tasks = [
                    str(x).strip()
                    for x in dataset.meta.episodes[episode]["tasks"]
                ]
                if instruction not in tasks:
                    raise RuntimeError(
                        f"t{task_id}_k{k} episode {episode} "
                        f"has task text {tasks}"
                    )

            target_frames[f"t{task_id}_k{k}"] = int(
                dataset.meta.total_frames
            )

    info = {
        "seen_train_root": str(SEEN_TRAIN_ROOT),
        "seen_train_episodes": int(
            seen_dataset.meta.total_episodes
        ),
        "seen_train_frames": int(
            seen_dataset.meta.total_frames
        ),
        "target_frames": target_frames,
        "target_manifest": str(manifest_path),
    }
    print(
        f"seen: {info['seen_train_episodes']} episodes, "
        f"{info['seen_train_frames']} frames"
    )
    print("target manifest:", manifest_path)
    return info


def replay_gripper_smoke(
    task_id: int,
    k: int = 5,
    episode: int = 0,
    *,
    top_init_candidates: int = 5,
    seed: int = 123,
) -> dict:
    import gymnasium as gym
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.envs.libero import create_libero_envs

    if task_id not in TARGETS:
        raise ValueError(task_id)

    dataset = LeRobotDataset(
        target_repo_id(task_id, k),
        root=subset_root(task_id, k),
    )
    if episode >= dataset.meta.total_episodes:
        raise ValueError(
            f"episode={episode}, dataset has {dataset.meta.total_episodes}"
        )

    start = int(dataset.meta.episodes[episode]["dataset_from_index"])
    stop = int(dataset.meta.episodes[episode]["dataset_to_index"])
    camera_key = "observation.images.image"

    first_frame = dataset[start][camera_key]
    if hasattr(first_frame, "detach"):
        first_frame = first_frame.detach().cpu().numpy()
    if first_frame.ndim == 3 and first_frame.shape[0] in (1, 3):
        first_frame = np.moveaxis(first_frame, 0, -1)
    if first_frame.dtype != np.uint8:
        if float(np.nanmax(first_frame)) <= 1.0:
            first_frame = first_frame * 255.0
        first_frame = np.clip(first_frame, 0, 255).astype(np.uint8)

    actions = np.stack(
        [
            np.asarray(dataset[index]["action"], dtype=np.float32)
            for index in range(start, stop)
        ]
    )
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise RuntimeError(f"Expected [T,7] actions, got {actions.shape}")

    def make_env():
        height, width = first_frame.shape[:2]
        envs = create_libero_envs(
            task="libero_goal",
            n_envs=1,
            gym_kwargs={
                "task_ids": [task_id],
                "observation_height": int(height),
                "observation_width": int(width),
            },
            init_states=True,
            env_cls=gym.vector.SyncVectorEnv,
            control_mode="relative",
        )
        return envs["libero_goal"][task_id]

    env = make_env()
    try:
        scores = []
        for init_state_id in range(len(env.envs[0]._init_states)):
            observation, _ = env.reset(seed=seed)
            env_frame = np.asarray(observation["pixels"]["image"][0], dtype=np.uint8)
            if env_frame.shape != first_frame.shape:
                raise RuntimeError(
                    f"Camera mismatch: env {env_frame.shape}, "
                    f"dataset {first_frame.shape}"
                )
            mse = float(
                np.mean(
                    (
                        env_frame.astype(np.float32)
                        - first_frame.astype(np.float32)
                    )
                    ** 2
                )
            )
            scores.append((mse, init_state_id))
    finally:
        env.close()

    for mse, init_state_id in sorted(scores)[:max(1, top_init_candidates)]:
        env = make_env()
        try:
            for _ in range(init_state_id + 1):
                env.reset(seed=seed)

            success = False
            used_steps = 0
            for used_steps, action in enumerate(actions, start=1):
                _, _, terminated, truncated, info = env.step(action[None, :])
                success_values = np.asarray(
                    info.get("is_success", [False])
                ).reshape(-1)
                success = bool(success_values[0]) if len(success_values) else False
                terminal = bool(np.asarray(terminated).reshape(-1)[0])
                timed_out = bool(np.asarray(truncated).reshape(-1)[0])
                if success or terminal or timed_out:
                    break

            if success:
                return {
                    "task_id": task_id,
                    "instruction": TARGETS[task_id],
                    "K": k,
                    "episode": episode,
                    "success": True,
                    "matched_init_state": int(init_state_id),
                    "steps": int(used_steps),
                    "first_frame_mse": mse,
                }
        finally:
            env.close()

    raise RuntimeError(
        f"Replay failed for task {task_id}. "
        "Check gripper conversion, task mapping and LIBERO setup."
    )
