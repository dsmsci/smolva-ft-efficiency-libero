from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(
    command: Iterable[str],
    *,
    cwd: str | Path | None = None,
    log_path: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    command = [str(x) for x in command]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    if log_path is None:
        return subprocess.run(command, cwd=cwd, env=merged_env, check=True)

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        return subprocess.run(
            command,
            cwd=cwd,
            env=merged_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def assert_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SmolVLA training/evaluation in this repository.")
    if not torch.backends.cudnn.is_available():
        raise RuntimeError("cuDNN is not available although CUDA is visible.")
