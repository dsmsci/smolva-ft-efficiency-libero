from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image


def load_reward_model(repo: Path, checkpoint: Path):
    sys.path.insert(0, str(repo))
    from models.clip_withhead import load

    model, _ = load(
        str(checkpoint),
        name="ViT-B/16",
        device="cuda",
        T=1,
        use_cache=False,
        bin_num=20,
        use_bin=True,
        diff_representation=False,
        use_text=False,
        implicit_negative=True,
    )
    model = model.cuda().eval()
    transform = transforms.Compose(
        [
            transforms.Resize(
                224,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )
    return model, transform


def video_reward(model, transform, path: Path) -> np.ndarray:
    import decord

    video = decord.VideoReader(str(path))
    chunks = []
    with torch.no_grad():
        for start in range(0, len(video), 64):
            low = max(0, start - 1)
            high = min(len(video), start + 64)
            frames = video.get_batch(list(range(low, high))).asnumpy()
            images = torch.stack(
                [transform(Image.fromarray(frame)) for frame in frames]
            ).cuda()
            reward = model.predict_progress(images, None).detach().float().cpu()
            if start:
                reward = reward[1:]
            chunks.append(reward)
            del images
    values = torch.cat(chunks).numpy().reshape(-1)
    if len(values) != len(video):
        raise RuntimeError(
            f"{path}: reward length {len(values)} != video length {len(video)}"
        )
    return values


def export_progress(args) -> None:
    model, transform = load_reward_model(args.repo, args.checkpoint)
    rows = []
    for video in sorted(args.videos.glob("episode_*.mp4")):
        episode = int(video.stem.split("_")[-1])
        reward = video_reward(model, transform, video)
        potential = np.cumsum(reward)
        span = float(potential.max() - potential.min())
        progress = (potential - potential.min()) / max(span, 1e-8)
        for frame_index, value in enumerate(progress):
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": frame_index,
                    "progress_sparse": float(value),
                }
            )
    pd.DataFrame(rows).to_csv(args.out, index=False)


def score_pool(args) -> None:
    model, transform = load_reward_model(args.repo, args.checkpoint)
    manifest = pd.read_csv(args.manifest)
    rows = []
    for row in manifest.itertuples(index=False):
        reward = video_reward(model, transform, Path(row.video))
        score = float(np.mean(reward[1:])) if len(reward) > 1 else float("nan")
        rows.append({"row_id": int(row.row_id), "reward_score": score})
    pd.DataFrame(rows).to_csv(args.out, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    progress = subparsers.add_parser("progress")
    progress.add_argument("--repo", type=Path, required=True)
    progress.add_argument("--checkpoint", type=Path, required=True)
    progress.add_argument("--videos", type=Path, required=True)
    progress.add_argument("--out", type=Path, required=True)
    progress.set_defaults(func=export_progress)

    pool = subparsers.add_parser("pool")
    pool.add_argument("--repo", type=Path, required=True)
    pool.add_argument("--checkpoint", type=Path, required=True)
    pool.add_argument("--manifest", type=Path, required=True)
    pool.add_argument("--out", type=Path, required=True)
    pool.set_defaults(func=score_pool)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
