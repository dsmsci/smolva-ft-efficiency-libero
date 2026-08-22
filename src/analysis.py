from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .settings import TARGETS


def _resolve_video(value: str | None, eval_json: Path) -> str | None:
    if not value:
        return None
    raw = Path(value)
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([eval_json.parent / raw, Path.cwd() / raw])
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def collect_eval_results(run_specs: list[dict]) -> pd.DataFrame:
    rows = []
    for source_spec in run_specs:
        spec = dict(source_spec)
        path = Path(spec.pop("eval_json"))
        info = json.loads(path.read_text(encoding="utf-8"))
        expected_task = spec.pop("expected_task_id", None)

        for task in info["per_task"]:
            task_id = int(task["task_id"])
            if expected_task is not None and task_id != int(expected_task):
                raise RuntimeError(f"{path}: expected task {expected_task}, found {task_id}")
            successes = task["metrics"]["successes"]
            videos = task["metrics"].get("video_paths", [])
            for episode, success in enumerate(successes):
                video = videos[episode] if episode < len(videos) else None
                rows.append(
                    {
                        **spec,
                        "task_id": task_id,
                        "episode": episode,
                        "success": int(bool(success)),
                        "video": _resolve_video(video, path),
                    }
                )
    return pd.DataFrame(rows)


def language_control(zero_shot: pd.DataFrame, wrong: pd.DataFrame, *, start_seed: int) -> pd.DataFrame:
    correct = zero_shot[
        zero_shot["method"].eq("zero_shot") & zero_shot["budget"].eq(0)
    ][["task_id", "episode", "success"]].copy()
    correct["condition"] = "correct"
    correct["instruction"] = correct["task_id"].map(TARGETS)
    correct["eval_seed"] = start_seed + correct["episode"].astype(int)
    correct = correct[["task_id", "condition", "instruction", "episode", "eval_seed", "success"]]

    keys = ["task_id", "eval_seed"]
    if correct.duplicated(keys).any() or wrong.duplicated(keys).any():
        raise RuntimeError("Language-control pairing keys are not unique")
    if set(map(tuple, correct[keys].to_numpy())) != set(map(tuple, wrong[keys].to_numpy())):
        raise RuntimeError("Correct and wrong-language arms do not use identical task/seed pairs")
    return pd.concat([correct, wrong[correct.columns]], ignore_index=True)


def language_control_summary(frame: pd.DataFrame) -> pd.DataFrame:
    paired = frame.pivot_table(
        index=["task_id", "eval_seed"], columns="condition", values="success", aggfunc="first"
    ).dropna()
    paired["delta"] = paired["correct"] - paired["wrong"]
    per_task = paired.groupby("task_id")[["correct", "wrong", "delta"]].mean().reset_index()
    macro = pd.DataFrame(
        [{
            "task_id": "macro",
            "correct": paired["correct"].mean(),
            "wrong": paired["wrong"].mean(),
            "delta": paired["delta"].mean(),
        }]
    )
    out = pd.concat([per_task, macro], ignore_index=True)
    macro_row = out[out["task_id"].astype(str).eq("macro")].iloc[0]
    if float(macro_row["correct"]) == 0.0 and float(macro_row["wrong"]) == 0.0:
        status = "floor: language control is inconclusive"
    elif float(macro_row["delta"]) > 0:
        status = "correct instruction performs better"
    else:
        status = "no positive language-control gap"
    out["interpretation"] = ""
    out.loc[out["task_id"].astype(str).eq("macro"), "interpretation"] = status
    return out


def attach_shared_zero_shot(raw: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    zero = raw[raw["budget"].eq(0) & raw["method"].eq("zero_shot")].copy()
    nonzero = raw[raw["budget"].ne(0)].copy()
    if zero.empty:
        raise ValueError("No zero-shot rows")
    anchors = []
    for method in methods:
        copy = zero.copy()
        copy["method"] = method
        copy["train_seed"] = np.nan
        anchors.append(copy)
    return pd.concat([nonzero, *anchors], ignore_index=True)


def success_rates(raw: pd.DataFrame) -> pd.DataFrame:
    keys = ["method", "budget", "train_seed", "task_id"]
    rows = []
    for values, group in raw.groupby(keys, dropna=False):
        successes = int(group["success"].sum())
        episodes = int(len(group))
        interval = binomtest(successes, episodes).proportion_ci(confidence_level=0.95, method="wilson")
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "successes": successes,
                "episodes": episodes,
                "success_rate": successes / episodes,
                "ci_low": float(interval.low),
                "ci_high": float(interval.high),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def cost_curve_summary(raw: pd.DataFrame) -> pd.DataFrame:
    cells = success_rates(raw)
    return (
        cells.groupby(["method", "budget"], as_index=False)
        .agg(
            mean_success=("success_rate", "mean"),
            std_success=("success_rate", "std"),
            cells=("success_rate", "size"),
        )
        .sort_values(["method", "budget"])
        .reset_index(drop=True)
    )


def failure_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    columns = ["method", "task_id", "budget", "train_seed", "episode", "video"]
    return (
        raw[raw["success"].eq(0) & raw["video"].notna()][columns]
        .sort_values(["method", "task_id", "budget", "train_seed", "episode"])
        .reset_index(drop=True)
    )
