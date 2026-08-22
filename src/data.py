from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .settings import (
    BUDGETS,
    DATA,
    DATASET_ID,
    FIXED,
    GOAL_REPO_ID,
    GOAL_TRAIN_ROOT,
    RAW,
    SEEN_REPO_ID,
    SEEN_TRAIN_ROOT,
    TARGETS,
    VIDEO_BACKEND,
    VIEWS,
)


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
        raise FileNotFoundError(f"Dataset suite was not downloaded correctly: {root}")
    return root


def _copy_for_action_fix(src: Path, dst: Path) -> None:
    """Copy metadata/data, hard-link videos when possible, and never mutate raw files."""

    def copy_file(source: str, destination: str):
        source_path = Path(source)
        if source_path.suffix.lower() == ".mp4":
            try:
                os.link(source, destination)
                return destination
            except OSError:
                pass
        return shutil.copy2(source, destination)

    shutil.copytree(src, dst, copy_function=copy_file)


def _convert_gripper_parquets(root: Path) -> None:
    """NVIDIA LIBERO: g_data in {0,1}, 1=open. Env: +1=close, -1=open."""
    parquets = sorted((root / "data").rglob("*.parquet"))
    if not parquets:
        raise RuntimeError(f"No parquet files found under {root / 'data'}")

    observed_min = np.inf
    observed_max = -np.inf

    for parquet in parquets:
        frame = pd.read_parquet(parquet)
        if "action" not in frame.columns:
            raise KeyError(f"{parquet}: action column is missing")

        converted = []
        for raw_action in frame["action"]:
            action = np.asarray(raw_action, dtype=np.float32).copy()
            if action.shape[-1] != 7:
                raise ValueError(f"{parquet}: expected action dim 7, got {action.shape}")
            gripper = np.asarray(action[..., -1], dtype=np.float32)
            observed_min = min(observed_min, float(np.min(gripper)))
            observed_max = max(observed_max, float(np.max(gripper)))
            if float(np.min(gripper)) < -1e-4 or float(np.max(gripper)) > 1.0001:
                raise ValueError(
                    f"{parquet}: source gripper must be in [0,1], got "
                    f"[{float(np.min(gripper)):.4f}, {float(np.max(gripper)):.4f}]. "
                    "Refusing to convert twice."
                )
            action[..., -1] = 1.0 - 2.0 * gripper
            converted.append(action.tolist())

        frame["action"] = converted
        frame.to_parquet(parquet, index=False)

    print(f"source gripper range: [{observed_min:.4f}, {observed_max:.4f}]")


def _prepare_fixed_suite(src: Path, dst: Path, repo_id: str) -> Path:
    """Create an action-fixed copy atomically; original videos stay in their codec."""
    marker = dst / ".gripper_env_convention"
    if marker.exists():
        return dst

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    if dst.exists():
        # An interrupted conversion is unsafe to resume because it may contain a
        # mixture of [0,1] and [-1,1] actions. Rebuild from immutable raw data.
        shutil.rmtree(dst)

    _copy_for_action_fix(src, tmp)
    _convert_gripper_parquets(tmp)

    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=tmp, video_backend=VIDEO_BACKEND)
    recompute_stats(dataset, skip_image_video=True)
    (tmp / ".gripper_env_convention").write_text(
        "g_env = 1 - 2 * g_data; source videos preserved\n",
        encoding="utf-8",
    )
    tmp.replace(dst)
    return dst


def _normalize_instruction(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _episodes_for_instruction(dataset, instruction: str) -> list[int]:
    wanted = _normalize_instruction(instruction)
    selected: list[int] = []
    for episode in range(int(dataset.meta.total_episodes)):
        tasks = dataset.meta.episodes[episode]["tasks"]
        if any(_normalize_instruction(task) == wanted for task in tasks):
            selected.append(episode)
    return selected


def target_episode_ids(task_id: int, k: int | None = None) -> list[int]:
    if task_id not in TARGETS:
        raise ValueError(f"Unknown target task_id={task_id}")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        GOAL_REPO_ID,
        root=GOAL_TRAIN_ROOT,
        video_backend=VIDEO_BACKEND,
    )
    episodes = _episodes_for_instruction(dataset, TARGETS[task_id])
    if len(episodes) < max(BUDGETS):
        raise RuntimeError(
            f"Target '{TARGETS[task_id]}' has only {len(episodes)} episodes; "
            f"need at least {max(BUDGETS)}."
        )
    return episodes if k is None else episodes[:k]


def seen_instructions() -> list[str]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        SEEN_REPO_ID,
        root=SEEN_TRAIN_ROOT,
        video_backend=VIDEO_BACKEND,
    )
    values: list[str] = []
    seen: set[str] = set()
    for episode in range(int(dataset.meta.total_episodes)):
        for task in dataset.meta.episodes[episode]["tasks"]:
            text = str(task).strip()
            key = _normalize_instruction(text)
            if key and key not in seen:
                seen.add(key)
                values.append(text)
    return values


def seen_sanity_tasks(n: int = 3) -> list[tuple[int, str]]:
    """Find benchmark task ids whose language strings really occur in libero_90."""
    from libero.libero import benchmark

    present = {_normalize_instruction(x) for x in seen_instructions()}
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    matches: list[tuple[int, str]] = []
    for task_id in range(int(suite.get_num_tasks())):
        task = suite.get_task(task_id)
        text = getattr(task, "language", None)
        if text is None and isinstance(task, dict):
            text = task.get("language")
        if text is None:
            continue
        text = str(text).strip()
        if _normalize_instruction(text) in present:
            matches.append((task_id, text))
        if len(matches) >= n:
            break
    if len(matches) < n:
        raise RuntimeError(
            f"Only {len(matches)} LIBERO-90 benchmark tasks could be matched to dataset text."
        )
    return matches



def _selected_numeric_stats(dataset) -> dict:
    """Compute normalization stats only from the episodes selected in ``dataset``."""
    from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats

    meta_keys = {"index", "episode_index", "task_index", "frame_index", "timestamp"}
    numeric_features = {
        key: feature
        for key, feature in dataset.meta.features.items()
        if feature["dtype"] not in {"image", "video", "string", "language"} and key not in meta_keys
    }
    if not numeric_features:
        return dict(dataset.meta.stats)

    hf = dataset.hf_dataset.with_format(None)
    episode_index = np.asarray(hf["episode_index"], dtype=np.int64)
    arrays: dict[str, np.ndarray] = {}
    for key in numeric_features:
        if key not in hf.column_names:
            continue
        values = hf[key]
        array = np.asarray(values)
        if array.dtype == object:
            array = np.stack([np.asarray(value) for value in values])
        arrays[key] = array

    selected = dataset.episodes
    if selected is None:
        selected = sorted(np.unique(episode_index).tolist())

    episode_stats = []
    for source_episode in selected:
        mask = episode_index == int(source_episode)
        if not bool(mask.any()):
            raise RuntimeError(f"Selected episode {source_episode} is absent from the loaded HF rows")
        episode_data = {key: array[mask] for key, array in arrays.items()}
        episode_stats.append(compute_episode_stats(episode_data, numeric_features))

    stats = aggregate_stats(episode_stats)
    # Image/video statistics are not recomputed. SmolVLA uses identity normalization
    # for visual inputs, but retaining the existing entries keeps the metadata complete.
    for key, value in dataset.meta.stats.items():
        if key not in stats:
            stats[key] = value
    return stats


def _make_target_view(task_id: int, k: int, chosen: list[int]) -> tuple[str, Path]:
    """Create a tiny metadata view with K-only normalization stats; data/videos are symlinked."""
    from lerobot.datasets.io_utils import write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = f"local/libero_goal_t{task_id}_k{k}"
    dst = VIEWS / f"libero_goal_t{task_id}_k{k}"
    marker = dst / ".selection"
    expected = ",".join(str(int(x)) for x in chosen)
    current_links = False
    if marker.exists():
        try:
            data_link = dst / "data"
            videos_link = dst / "videos"
            data_ok = data_link.exists() and data_link.resolve() == (GOAL_TRAIN_ROOT / "data").resolve()
            source_videos = GOAL_TRAIN_ROOT / "videos"
            videos_ok = (
                (not source_videos.exists() and not videos_link.exists())
                or (videos_link.exists() and videos_link.resolve() == source_videos.resolve())
            )
            current_links = data_ok and videos_ok
        except OSError:
            current_links = False
    if (
        marker.exists()
        and marker.read_text(encoding="utf-8").strip() == expected
        and current_links
    ):
        return repo_id, dst

    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(dst, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    shutil.copytree(GOAL_TRAIN_ROOT / "meta", tmp / "meta")
    for name in ("data", "videos"):
        source = (GOAL_TRAIN_ROOT / name).resolve()
        if source.exists():
            (tmp / name).symlink_to(source, target_is_directory=True)

    selected = LeRobotDataset(
        GOAL_REPO_ID,
        root=GOAL_TRAIN_ROOT,
        episodes=chosen,
        video_backend=VIDEO_BACKEND,
    )
    stats = _selected_numeric_stats(selected)
    write_stats(stats, tmp)
    (tmp / ".selection").write_text(expected + "\n", encoding="utf-8")
    tmp.replace(dst)

    check = LeRobotDataset(repo_id, root=dst, episodes=chosen, video_backend=VIDEO_BACKEND)
    if [int(x) for x in check.episodes] != [int(x) for x in chosen]:
        raise RuntimeError(f"Target view t{task_id}/k{k} did not preserve source episode ids")
    return repo_id, dst


def target_training_source(task_id: int, k: int) -> tuple[str, Path, list[int]]:
    """Return the leak-free metadata view used for target fine-tuning."""
    chosen = target_episode_ids(task_id, k)
    repo_id, root = _make_target_view(task_id, k, chosen)
    return repo_id, root, chosen

def _validate_selected_episodes(task_id: int, chosen: list[int]) -> int:
    """Use the exact LeRobot 0.6.1 episode filter and verify the result."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        GOAL_REPO_ID,
        root=GOAL_TRAIN_ROOT,
        episodes=chosen,
        video_backend=VIDEO_BACKEND,
    )
    loaded = [int(x) for x in dataset.episodes]
    if loaded != [int(x) for x in chosen]:
        raise RuntimeError(
            f"t{task_id}: LeRobot loaded episodes {loaded}, requested {chosen}. "
            "Do not train until the dataset version/filtering is fixed."
        )
    if int(dataset.num_episodes) != len(chosen):
        raise RuntimeError(
            f"t{task_id}: selected dataset reports {dataset.num_episodes} episodes, "
            f"expected {len(chosen)}."
        )
    return int(dataset.num_frames)


def _decode_smoke(repo_id: str, root: Path, episode: int = 0) -> None:
    """Decode real video tensors before any expensive training starts."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=[episode],
        video_backend=VIDEO_BACKEND,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"Decode smoke dataset is empty: {root}")
    sample = dataset[0]
    missing = [key for key in dataset.meta.video_keys if key not in sample]
    if missing:
        raise RuntimeError(f"Video decoder did not return keys: {missing}")
    for key in dataset.meta.video_keys:
        tensor = sample[key]
        shape = tuple(getattr(tensor, "shape", ()))
        if len(shape) != 3:
            raise RuntimeError(f"Decoded {key} has unexpected shape {shape}")


def prepare_data(force: bool = False) -> dict:
    """Download, fix gripper convention, verify AV1 decoding and exact K subsets."""
    if force:
        shutil.rmtree(FIXED, ignore_errors=True)
        shutil.rmtree(VIEWS, ignore_errors=True)

    RAW.mkdir(parents=True, exist_ok=True)
    FIXED.mkdir(parents=True, exist_ok=True)
    VIEWS.mkdir(parents=True, exist_ok=True)

    seen_raw = _download_suite("libero_90")
    goal_raw = _download_suite("libero_goal")
    _prepare_fixed_suite(seen_raw, SEEN_TRAIN_ROOT, SEEN_REPO_ID)
    _prepare_fixed_suite(goal_raw, GOAL_TRAIN_ROOT, GOAL_REPO_ID)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    seen = LeRobotDataset(
        SEEN_REPO_ID,
        root=SEEN_TRAIN_ROOT,
        video_backend=VIDEO_BACKEND,
    )
    goal = LeRobotDataset(
        GOAL_REPO_ID,
        root=GOAL_TRAIN_ROOT,
        video_backend=VIDEO_BACKEND,
    )

    # This catches unsupported/broken video decoding before training.
    _decode_smoke(SEEN_REPO_ID, SEEN_TRAIN_ROOT, episode=0)
    _decode_smoke(GOAL_REPO_ID, GOAL_TRAIN_ROOT, episode=0)

    manifest_rows: list[dict] = []
    target_frames: dict[str, int] = {}
    for task_id, instruction in TARGETS.items():
        all_ids = _episodes_for_instruction(goal, instruction)
        if len(all_ids) < max(BUDGETS):
            raise RuntimeError(f"{instruction}: only {len(all_ids)} episodes")
        for k in BUDGETS:
            chosen = all_ids[:k]
            frames = _validate_selected_episodes(task_id, chosen)
            view_repo_id, view_root = _make_target_view(task_id, k, chosen)
            target_frames[f"t{task_id}_k{k}"] = frames
            for rank, source_episode in enumerate(chosen):
                manifest_rows.append(
                    {
                        "task_id": task_id,
                        "instruction": instruction,
                        "K": k,
                        "order_in_budget": rank,
                        "source_episode": int(source_episode),
                        "training_repo_id": view_repo_id,
                        "training_root": str(view_root),
                    }
                )

    manifest_path = DATA / "target_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    info = {
        "seen_train_root": str(SEEN_TRAIN_ROOT),
        "seen_train_episodes": int(seen.meta.total_episodes),
        "seen_train_frames": int(seen.meta.total_frames),
        "goal_episodes": int(goal.meta.total_episodes),
        "target_frames": target_frames,
        "target_manifest": str(manifest_path),
        "video_backend": VIDEO_BACKEND,
        "video_conversion": "none",
    }
    print(
        f"seen: {info['seen_train_episodes']} episodes, "
        f"{info['seen_train_frames']} frames; video backend={VIDEO_BACKEND}"
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
    """Replay one *dataset* demonstration in LIBERO; must reach success=1."""
    import gymnasium as gym
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.envs.libero import create_libero_envs

    chosen = target_episode_ids(task_id, k)
    if not 0 <= episode < len(chosen):
        raise ValueError(f"episode={episode}, selected budget has {len(chosen)} episodes")
    source_episode = chosen[episode]

    dataset = LeRobotDataset(
        GOAL_REPO_ID,
        root=GOAL_TRAIN_ROOT,
        video_backend=VIDEO_BACKEND,
    )
    meta = dataset.meta.episodes[source_episode]
    start = int(meta["dataset_from_index"])
    stop = int(meta["dataset_to_index"])
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
        [np.asarray(dataset[index]["action"], dtype=np.float32) for index in range(start, stop)]
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
        base_env = env.envs[0]
        if base_env._init_states is None:
            raise RuntimeError("LIBERO init states were not loaded for replay smoke")
        for init_state_id in range(len(base_env._init_states)):
            # LeRobot 0.6.1 exposes init_state_id specifically to choose the next
            # LIBERO initial state. Set it explicitly instead of depending on the
            # number of previous reset() calls.
            base_env.init_state_id = int(init_state_id)
            observation, _ = env.reset(seed=seed)
            env_frame = np.asarray(observation["pixels"]["image"][0], dtype=np.uint8)
            if env_frame.shape != first_frame.shape:
                raise RuntimeError(
                    f"Camera mismatch: env {env_frame.shape}, dataset {first_frame.shape}"
                )
            mse = float(
                np.mean((env_frame.astype(np.float32) - first_frame.astype(np.float32)) ** 2)
            )
            scores.append((mse, init_state_id))
    finally:
        env.close()

    for mse, init_state_id in sorted(scores)[: max(1, top_init_candidates)]:
        env = make_env()
        try:
            env.envs[0].init_state_id = int(init_state_id)
            env.reset(seed=seed)

            success = False
            used_steps = 0
            for used_steps, action in enumerate(actions, start=1):
                _, _, terminated, truncated, info = env.step(action[None, :])
                success_values = np.asarray(info.get("is_success", [False])).reshape(-1)
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
                    "source_episode": int(source_episode),
                    "success": True,
                    "matched_init_state": int(init_state_id),
                    "steps": int(used_steps),
                    "first_frame_mse": mse,
                }
        finally:
            env.close()

    raise RuntimeError(
        f"Replay failed for task {task_id}. Check gripper conversion, task mapping and LIBERO setup."
    )
