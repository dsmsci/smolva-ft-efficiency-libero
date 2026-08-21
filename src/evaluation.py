from __future__ import annotations

import gc
import json
from pathlib import Path

import pandas as pd
import torch

from .settings import ENV_RENAME_MAP, OUTPUTS, TARGETS
from .utils import run

WRONG_LANGUAGE = {
    0: TARGETS[1],
    1: TARGETS[2],
    2: TARGETS[0],
}


def eval_checkpoint(
    checkpoint: str | Path,
    tag: str,
    task_ids: tuple[int, ...],
    *,
    n_episodes: int = 20,
    seed: int = 10_000,
) -> Path:
    output = OUTPUTS / "eval" / tag
    result = output / "eval_info.json"
    if result.exists():
        return result

    run(
        [
            "lerobot-eval",
            f"--policy.path={checkpoint}",
            "--policy.device=cuda",
            f"--rename_map={json.dumps(ENV_RENAME_MAP)}",
            "--env.type=libero",
            "--env.task=libero_goal",
            f"--env.task_ids={json.dumps(list(task_ids))}",
            "--env.control_mode=relative",
            "--env.init_states=true",
            "--env.hard_reset=true",
            "--eval.batch_size=1",
            f"--eval.n_episodes={n_episodes}",
            "--eval.use_async_envs=false",
            f"--seed={seed}",
            f"--output_dir={output}",
        ],
        log_path=OUTPUTS / "logs" / f"eval__{tag}.log",
    )
    if not result.exists():
        raise FileNotFoundError(result)
    return result


def evaluate_grid(
    checkpoints: dict[tuple[int, int, int], Path],
    method: str,
    *,
    n_episodes: int = 20,
    seed: int = 10_000,
) -> list[dict]:
    specs = []
    for (task_id, k, train_seed), checkpoint in checkpoints.items():
        tag = f"{method}_t{task_id}_k{k}_s{train_seed}"
        result = eval_checkpoint(
            checkpoint,
            tag,
            (task_id,),
            n_episodes=n_episodes,
            seed=seed,
        )
        specs.append(
            {
                "eval_json": str(result),
                "method": method,
                "budget": k,
                "train_seed": train_seed,
                "expected_task_id": task_id,
            }
        )
    return specs


def eval_wrong_language(
    checkpoint: str | Path,
    *,
    n_episodes: int = 20,
    seed: int = 10_000,
) -> pd.DataFrame:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.factory import make_env, make_env_pre_post_processors
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.scripts.lerobot_eval import eval_policy

    output = OUTPUTS / "eval" / "wrong_language.csv"
    if output.exists():
        return pd.read_csv(output)

    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.pretrained_path = Path(checkpoint)
    config.device = "cuda"
    config.empty_cameras = 1

    rows = []
    for task_id, wrong_text in WRONG_LANGUAGE.items():
        env_config = LiberoEnv(
            task="libero_goal",
            task_ids=[task_id],
            init_states=True,
            hard_reset=True,
            control_mode="relative",
        )
        policy = make_policy(
            cfg=config,
            env_cfg=env_config,
            rename_map=ENV_RENAME_MAP,
        )
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=config.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": "cuda"},
                "rename_observations_processor": {
                    "rename_map": ENV_RENAME_MAP,
                },
            },
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_config,
            config,
        )
        envs = make_env(
            env_config,
            n_envs=1,
            use_async_envs=False,
        )
        env = next(iter(next(iter(envs.values())).values()))

        def wrong_preprocessor(observation):
            patched = dict(observation)
            patched["task"] = [wrong_text] * len(
                patched.get("task", [None])
            )
            return preprocessor(patched)

        info = eval_policy(
            env=env,
            policy=policy,
            env_preprocessor=env_preprocessor,
            env_postprocessor=env_postprocessor,
            preprocessor=wrong_preprocessor,
            postprocessor=postprocessor,
            n_episodes=n_episodes,
            max_episodes_rendered=0,
            start_seed=seed,
        )
        env.close()

        for episode in info["per_episode"]:
            rows.append(
                {
                    "task_id": task_id,
                    "condition": "wrong",
                    "instruction": wrong_text,
                    "episode": int(episode["episode_ix"]),
                    "eval_seed": int(episode["seed"]),
                    "success": int(bool(episode["success"])),
                }
            )

        del (
            policy,
            preprocessor,
            postprocessor,
            env_preprocessor,
            env_postprocessor,
        )
        gc.collect()
        torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
