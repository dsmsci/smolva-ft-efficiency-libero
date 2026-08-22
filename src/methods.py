from __future__ import annotations

import gc
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .data import seen_instructions, target_training_source
from .settings import (
    BUDGETS,
    OUTPUTS,
    SEEN_REPO_ID,
    SEEN_TRAIN_ROOT,
    TARGETS,
    TARGET_SEEDS,
)
from .training import (
    configure_optimizer,
    language_embedding,
    load_policy_and_processors,
    make_loader,
    make_preprocessor,
    next_batch,
    policy_loss_with_action_hidden,
    restore_resume,
    save_final,
    save_resume,
    select_video_time,
    setup_training,
    temporal_dataset,
    unwrap_policy,
    visual_embedding,
)


@dataclass(frozen=True)
class H1Config:
    horizon: int = 10
    clusters: int = 64
    max_pairs: int = 4096
    kmeans_iters: int = 15
    lambda_dyn: float = 0.10
    cosine_weight: float = 0.25


class VideoDynamicsHead(nn.Module):
    def __init__(self, action_hidden_dim: int, visual_dim: int, clusters: int):
        super().__init__()
        hidden = 256
        self.trunk = nn.Sequential(
            nn.Linear(action_hidden_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.classifier = nn.Linear(hidden, clusters)
        self.delta = nn.Linear(hidden, visual_dim)

    def forward(self, action_hidden: torch.Tensor):
        pooled = action_hidden.float().mean(dim=1)
        h = self.trunk(pooled)
        return self.classifier(h), self.delta(h)


def _checkpoint_signature(checkpoint: str | Path) -> tuple[tuple[str, int, int], ...]:
    """Cheap local signature used only to avoid reusing an H1 prior from different weights."""
    root = Path(checkpoint)
    if not root.exists():
        return ((str(checkpoint), -1, -1),)
    files = []
    for pattern in ("*.safetensors", "*.bin", "config.json"):
        files.extend(root.rglob(pattern))
    signature = []
    for file in sorted(set(files)):
        stat = file.stat()
        signature.append((str(file.relative_to(root)), int(stat.st_size), int(stat.st_mtime_ns)))
    if not signature:
        stat = root.stat()
        signature.append((".", int(stat.st_size), int(stat.st_mtime_ns)))
    return tuple(signature)


class InstructionAttentionHead(nn.Module):
    def __init__(self, language_dim: int, action_hidden_dim: int, action_dim: int):
        super().__init__()
        self.query = nn.Linear(language_dim, action_hidden_dim, bias=False)
        self.readout = nn.Sequential(
            nn.Linear(action_hidden_dim, 256),
            nn.GELU(),
            nn.Linear(256, action_dim),
        )

    def predict(self, action_hidden: torch.Tensor, language: torch.Tensor) -> torch.Tensor:
        action_hidden = action_hidden.float()
        q = self.query(language.float())
        scores = torch.einsum("btd,bd->bt", action_hidden, q) / (action_hidden.shape[-1] ** 0.5)
        weights = scores.softmax(dim=-1)
        pooled = torch.einsum("bt,btd->bd", weights, action_hidden)
        return self.readout(pooled)


def _without_actions(batch: dict) -> dict:
    """Remove action tensors and action padding; H1 prior must be action-free."""
    return {
        key: value
        for key, value in batch.items()
        if key != "action" and not key.startswith("action_")
    }


def _kmeans_cosine(x: torch.Tensor, k: int, iters: int, seed: int) -> torch.Tensor:
    if x.shape[0] < k:
        raise RuntimeError(f"Need at least {k} video transitions, got {x.shape[0]}")
    generator = torch.Generator(device=x.device).manual_seed(seed)
    centers = x[torch.randperm(x.shape[0], generator=generator, device=x.device)[:k]].clone()
    centers = F.normalize(centers, dim=-1)
    for _ in range(iters):
        labels = (x @ centers.T).argmax(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, labels, x)
        counts = torch.bincount(labels, minlength=k).to(x.dtype).unsqueeze(1)
        nonempty = counts.squeeze(1) > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty]
        centers = F.normalize(centers, dim=-1)
    return centers


def build_h1_video_prior(
    seen_checkpoint: str | Path,
    *,
    batch_size: int = 16,
    seed: int = 7,
    config: H1Config = H1Config(),
) -> Path:
    """Bonus A: derive transition prototypes from seen video without using action values."""
    from lerobot.configs.policies import PreTrainedConfig

    output = OUTPUTS / "priors" / "h1_video_dynamics.pt"
    expected_signature = _checkpoint_signature(seen_checkpoint)
    if output.exists():
        cached = torch.load(output, map_location="cpu", weights_only=False)
        same_config = (
            int(cached.get("horizon", -1)) == int(config.horizon)
            and int(cached.get("clusters", -1)) == int(config.clusters)
            and int(cached.get("max_pairs", -1)) == int(config.max_pairs)
            and int(cached.get("kmeans_iters", -1)) == int(config.kmeans_iters)
            and int(cached.get("seed", -1)) == int(seed)
            and cached.get("checkpoint_signature") == expected_signature
            and cached.get("uses_actions") is False
        )
        if same_config:
            return output
        output.unlink()

    setup_training(seed)
    policy_cfg = PreTrainedConfig.from_pretrained(seen_checkpoint)
    dataset = temporal_dataset(
        SEEN_REPO_ID,
        SEEN_TRAIN_ROOT,
        policy_cfg,
        video_horizon=config.horizon,
    )
    policy, preprocessor, _ = load_policy_and_processors(seen_checkpoint, dataset)
    policy.cuda().eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)

    loader = make_loader(
        dataset,
        batch_size,
        seed,
        drop_n_last_frames=config.horizon,
    )
    deltas = []
    count = 0
    with torch.no_grad():
        for raw in loader:
            # Bonus A path: drop action values before preprocessing; they are not targets, inputs, or clustering features.
            video_only = _without_actions(raw)
            current = preprocessor(select_video_time(video_only, 0))
            future = preprocessor(select_video_time(video_only, 1))
            z0 = visual_embedding(policy, current)
            z1 = visual_embedding(policy, future)
            delta = F.normalize((z1 - z0).float(), dim=-1)
            deltas.append(delta.detach())
            count += delta.shape[0]
            if count >= config.max_pairs:
                break

    x = torch.cat(deltas, dim=0)[: config.max_pairs]
    centers = _kmeans_cosine(x, config.clusters, config.kmeans_iters, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".tmp")
    torch.save(
        {
            "centers": centers.cpu(),
            "horizon": config.horizon,
            "clusters": config.clusters,
            "pairs": int(x.shape[0]),
            "max_pairs": int(config.max_pairs),
            "kmeans_iters": int(config.kmeans_iters),
            "seed": int(seed),
            "visual_dim": int(x.shape[1]),
            "seen_checkpoint": str(seen_checkpoint),
            "checkpoint_signature": expected_signature,
            "uses_actions": False,
        },
        tmp,
    )
    tmp.replace(output)
    del policy, preprocessor, dataset, loader, deltas, x, centers
    gc.collect()
    torch.cuda.empty_cache()
    return output


def _run_h1(
    seen_checkpoint: str | Path,
    prior_path: str | Path,
    task_id: int,
    k: int,
    seed: int,
    *,
    steps: int,
    batch_size: int,
    lambda_dyn: float = 0.10,
    cosine_weight: float = 0.25,
) -> Path:
    from lerobot.configs.policies import PreTrainedConfig

    run_root = OUTPUTS / "target" / "h1" / f"t{task_id}" / f"k{k}" / f"seed_{seed}"
    if (run_root / "final").exists():
        return run_root / "final"

    setup_training(seed)
    prior = torch.load(prior_path, map_location="cpu", weights_only=False)
    horizon = int(prior["horizon"])
    target_repo_id, target_root, episodes = target_training_source(task_id, k)
    cfg = PreTrainedConfig.from_pretrained(seen_checkpoint)
    dataset = temporal_dataset(
        target_repo_id,
        target_root,
        cfg,
        episodes=episodes,
        video_horizon=horizon,
    )
    policy, preprocessor, postprocessor = load_policy_and_processors(seen_checkpoint, dataset)
    policy.cuda().train()
    base = unwrap_policy(policy)
    action_hidden_dim = int(base.model.action_out_proj.in_features)
    centers = F.normalize(prior["centers"].cuda().float(), dim=-1)
    head = VideoDynamicsHead(action_hidden_dim, int(prior["visual_dim"]), centers.shape[0]).cuda()

    trainable, optimizer, scheduler, grad_clip = configure_optimizer(policy, steps, head.parameters())
    start = restore_resume(run_root, policy, optimizer, scheduler, aux_module=head)
    loader = make_loader(dataset, batch_size, seed, start_step=start, drop_n_last_frames=horizon)
    iterator = iter(loader)
    log = OUTPUTS / "logs" / f"h1__t{task_id}__k{k}__seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    for step in range(start + 1, steps + 1):
        raw, iterator = next_batch(iterator, loader)
        current = preprocessor(select_video_time(raw, 0))
        future = preprocessor(select_video_time(raw, 1))
        with torch.no_grad():
            z0 = visual_embedding(policy, current)
            z1 = visual_embedding(policy, future)
            delta = F.normalize((z1 - z0).float(), dim=-1)
            labels = (delta @ centers.T).argmax(dim=1)

        bc, hidden = policy_loss_with_action_hidden(policy, current)
        logits, predicted_delta = head(hidden)
        dyn = F.cross_entropy(logits, labels)
        dyn = dyn + cosine_weight * (1.0 - F.cosine_similarity(predicted_delta, delta, dim=-1)).mean()
        loss = bc + lambda_dyn * dyn

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            message = f"H1 step={step} total={float(loss):.5f} bc={float(bc):.5f} dyn={float(dyn):.5f}"
            print(message)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        if step % 500 == 0 and step < steps:
            save_resume(run_root, step, policy, optimizer, scheduler, aux_module=head)

    final = save_final(run_root, policy, preprocessor, postprocessor)
    del policy, head, loader, iterator, dataset
    gc.collect(); torch.cuda.empty_cache()
    return final


def _negative_seen_texts(task_text: str) -> list[str]:
    candidates = [x for x in seen_instructions() if x.strip().lower() != task_text.strip().lower()]
    if not candidates:
        raise RuntimeError("No unrelated seen instruction is available for H2")
    return candidates


def _run_h2(
    seen_checkpoint: str | Path,
    task_id: int,
    k: int,
    seed: int,
    *,
    steps: int,
    batch_size: int,
    lambda_instruction: float = 0.10,
    margin: float = 0.10,
) -> Path:
    from lerobot.configs.policies import PreTrainedConfig

    run_root = OUTPUTS / "target" / "h2" / f"t{task_id}" / f"k{k}" / f"seed_{seed}"
    if (run_root / "final").exists():
        return run_root / "final"

    setup_training(seed)
    cfg = PreTrainedConfig.from_pretrained(seen_checkpoint)
    target_repo_id, target_root, episodes = target_training_source(task_id, k)
    dataset = temporal_dataset(
        target_repo_id,
        target_root,
        cfg,
        episodes=episodes,
    )
    policy, preprocessor, postprocessor = load_policy_and_processors(seen_checkpoint, dataset)
    policy.cuda().train()
    base = unwrap_policy(policy)
    lang_dim = int(base.model.vlm_with_expert.config.text_config.hidden_size)
    hidden_dim = int(base.model.action_out_proj.in_features)
    action_dim = int(base.config.output_features["action"].shape[0])
    head = InstructionAttentionHead(lang_dim, hidden_dim, action_dim).cuda()

    trainable, optimizer, scheduler, grad_clip = configure_optimizer(policy, steps, head.parameters())
    start = restore_resume(run_root, policy, optimizer, scheduler, aux_module=head)
    loader = make_loader(dataset, batch_size, seed, start_step=start)
    iterator = iter(loader)
    negative_texts = _negative_seen_texts(TARGETS[task_id])
    log = OUTPUTS / "logs" / f"h2__t{task_id}__k{k}__seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    for step in range(start + 1, steps + 1):
        raw, iterator = next_batch(iterator, loader)
        positive_raw = dict(raw)
        negative_raw = dict(raw)
        processed = preprocessor(positive_raw)
        batch_n = int(next(v.shape[0] for v in raw.values() if hasattr(v, "shape") and v.ndim > 0))
        negative_text = negative_texts[(seed + step) % len(negative_texts)]
        negative_raw["task"] = [negative_text] * batch_n
        negative_processed = preprocessor(negative_raw)

        with torch.no_grad():
            q_pos = language_embedding(policy, processed)
            q_neg = language_embedding(policy, negative_processed)
            target_chunk = base.prepare_action(processed)[..., :action_dim]
            action_pad = processed.get("action_is_pad")
            if action_pad is None:
                target_action = target_chunk.mean(dim=1)
            else:
                valid = (~action_pad.bool()).to(target_chunk.dtype).unsqueeze(-1)
                target_action = (target_chunk * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        bc, hidden = policy_loss_with_action_hidden(policy, processed)
        pred_pos = head.predict(hidden, q_pos)
        pred_neg = head.predict(hidden, q_neg)
        pos = F.smooth_l1_loss(pred_pos, target_action, reduction="none").mean(dim=1)
        neg = F.smooth_l1_loss(pred_neg, target_action, reduction="none").mean(dim=1)
        aux = pos.mean() + 0.5 * F.relu(margin + pos - neg).mean()
        loss = bc + lambda_instruction * aux

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step(); scheduler.step()

        if step % 50 == 0:
            message = f"H2 step={step} total={float(loss):.5f} bc={float(bc):.5f} instr={float(aux):.5f}"
            print(message)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        if step % 500 == 0 and step < steps:
            save_resume(run_root, step, policy, optimizer, scheduler, aux_module=head)

    final = save_final(run_root, policy, preprocessor, postprocessor)
    del policy, head, loader, iterator, dataset
    gc.collect(); torch.cuda.empty_cache()
    return final


def _run_h3(
    seen_checkpoint: str | Path,
    task_id: int,
    k: int,
    seed: int,
    *,
    steps: int,
    batch_size: int,
    lambda_sp: float = 10000.0,
) -> Path:
    """L2-SP: anchor trainable SmolVLA weights to the seen checkpoint."""
    from lerobot.configs.policies import PreTrainedConfig

    run_root = OUTPUTS / "target" / "h3" / f"t{task_id}" / f"k{k}" / f"seed_{seed}"
    if (run_root / "final").exists():
        return run_root / "final"
    setup_training(seed)
    cfg = PreTrainedConfig.from_pretrained(seen_checkpoint)
    target_repo_id, target_root, episodes = target_training_source(task_id, k)
    dataset = temporal_dataset(target_repo_id, target_root, cfg, episodes=episodes)
    policy, preprocessor, postprocessor = load_policy_and_processors(seen_checkpoint, dataset)
    policy.cuda().train()
    policy_trainable = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    reference = {n: p.detach().clone() for n, p in policy_trainable}
    trainable, optimizer, scheduler, grad_clip = configure_optimizer(policy, steps)
    start = restore_resume(run_root, policy, optimizer, scheduler)
    loader = make_loader(dataset, batch_size, seed, start_step=start)
    iterator = iter(loader)
    log = OUTPUTS / "logs" / f"h3__t{task_id}__k{k}__seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    for step in range(start + 1, steps + 1):
        raw, iterator = next_batch(iterator, loader)
        processed = preprocessor(raw)
        bc, _ = policy(processed)
        squared = torch.stack([(p.float() - reference[n].float()).pow(2).sum() for n, p in policy_trainable]).sum()
        total_params = sum(p.numel() for _, p in policy_trainable)
        sp = squared / float(total_params)
        loss = bc + lambda_sp * sp
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step(); scheduler.step()
        if step % 50 == 0:
            message = f"H3 step={step} total={float(loss):.5f} bc={float(bc):.5f} l2sp={float(sp):.7f}"
            print(message)
            with log.open("a", encoding="utf-8") as handle: handle.write(message + "\n")
        if step % 500 == 0 and step < steps:
            save_resume(run_root, step, policy, optimizer, scheduler)

    final = save_final(run_root, policy, preprocessor, postprocessor)
    del policy, loader, iterator, dataset, reference
    gc.collect(); torch.cuda.empty_cache()
    return final


def _run_h4(
    seen_checkpoint: str | Path,
    task_id: int,
    k: int,
    seed: int,
    *,
    steps: int,
    batch_size: int,
    replay_interval: int = 4,
    lambda_seen: float = 0.5,
) -> Path:
    """Rehearsal: add a seen BC batch every replay_interval target steps."""
    from lerobot.configs.policies import PreTrainedConfig

    run_root = OUTPUTS / "target" / "h4" / f"t{task_id}" / f"k{k}" / f"seed_{seed}"
    if (run_root / "final").exists(): return run_root / "final"
    setup_training(seed)
    cfg = PreTrainedConfig.from_pretrained(seen_checkpoint)
    target_repo_id, target_root, episodes = target_training_source(task_id, k)
    target_ds = temporal_dataset(target_repo_id, target_root, cfg, episodes=episodes)
    seen_ds = temporal_dataset(SEEN_REPO_ID, SEEN_TRAIN_ROOT, cfg)
    policy, target_pre, postprocessor = load_policy_and_processors(seen_checkpoint, target_ds)
    # Keep one normalized action/state coordinate system throughout target adaptation.
    # Seen replay is allowed by the task, but its own global stats must not redefine
    # the policy output coordinates inside the same optimization run.
    seen_pre = make_preprocessor(
        seen_checkpoint,
        policy,
        seen_ds,
        normalization_stats=target_ds.meta.stats,
    )
    policy.cuda().train()
    trainable, optimizer, scheduler, grad_clip = configure_optimizer(policy, steps)
    start = restore_resume(run_root, policy, optimizer, scheduler)
    target_loader = make_loader(target_ds, batch_size, seed, start_step=start)
    seen_loader = make_loader(seen_ds, batch_size, seed + 10000, start_step=start // replay_interval)
    target_it, seen_it = iter(target_loader), iter(seen_loader)
    log = OUTPUTS / "logs" / f"h4__t{task_id}__k{k}__seed_{seed}.log"; log.parent.mkdir(parents=True, exist_ok=True)

    for step in range(start + 1, steps + 1):
        raw, target_it = next_batch(target_it, target_loader)
        bc, _ = policy(target_pre(raw))
        seen_loss = torch.zeros((), device=bc.device)
        if step % replay_interval == 0:
            seen_raw, seen_it = next_batch(seen_it, seen_loader)
            seen_loss, _ = policy(seen_pre(seen_raw))
        loss = bc + lambda_seen * seen_loss
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step(); scheduler.step()
        if step % 50 == 0:
            message = f"H4 step={step} total={float(loss):.5f} bc={float(bc):.5f} seen={float(seen_loss):.5f}"
            print(message)
            with log.open("a", encoding="utf-8") as handle: handle.write(message + "\n")
        if step % 500 == 0 and step < steps:
            save_resume(run_root, step, policy, optimizer, scheduler)

    final = save_final(run_root, policy, target_pre, postprocessor)
    del policy, target_loader, seen_loader, target_ds, seen_ds
    gc.collect(); torch.cuda.empty_cache()
    return final



def smoke_custom_objectives(
    seen_checkpoint: str | Path,
    prior_path: str | Path,
    *,
    task_id: int = 0,
    k: int = 5,
    batch_size: int = 2,
    seed: int = 31415,
) -> dict[str, float | int]:
    """Run one real forward/backward pass for H1-H4 before the expensive grids.

    No optimizer step is taken and no checkpoint is written. This is an API/data
    smoke test: it exercises the same dataset filtering, processors, hidden-state
    hook and auxiliary objectives used by the four custom training loops.
    """
    from lerobot.configs.policies import PreTrainedConfig

    setup_training(seed)
    prior = torch.load(prior_path, map_location="cpu", weights_only=False)
    if bool(prior.get("uses_actions", True)):
        raise RuntimeError("H1 prior is not marked action-free")
    horizon = int(prior["horizon"])

    target_repo_id, target_root, episodes = target_training_source(task_id, k)
    cfg = PreTrainedConfig.from_pretrained(seen_checkpoint)
    target_ds = temporal_dataset(
        target_repo_id,
        target_root,
        cfg,
        episodes=episodes,
        video_horizon=horizon,
    )
    policy, target_pre, _ = load_policy_and_processors(seen_checkpoint, target_ds)
    policy.cuda().train()
    base = unwrap_policy(policy)
    target_loader = make_loader(
        target_ds,
        batch_size,
        seed,
        drop_n_last_frames=horizon,
    )
    raw, _ = next_batch(iter(target_loader), target_loader)

    def backward_check(name: str, loss: torch.Tensor, extra: nn.Module | None = None) -> float:
        if loss.ndim != 0 or not bool(torch.isfinite(loss.detach()).item()):
            raise RuntimeError(f"{name} smoke produced invalid loss: {loss}")
        policy.zero_grad(set_to_none=True)
        if extra is not None:
            extra.zero_grad(set_to_none=True)
        loss.backward()
        parameters = [p for p in policy.parameters() if p.requires_grad]
        if extra is not None:
            parameters += [p for p in extra.parameters() if p.requires_grad]
        finite_grads = [p.grad for p in parameters if p.grad is not None and torch.isfinite(p.grad).all()]
        if not finite_grads:
            raise RuntimeError(f"{name} smoke produced no finite gradients")
        value = float(loss.detach().cpu())
        policy.zero_grad(set_to_none=True)
        if extra is not None:
            extra.zero_grad(set_to_none=True)
        return value

    # H1: transition prototype + direction from target visual transition.
    current = target_pre(select_video_time(raw, 0))
    future = target_pre(select_video_time(raw, 1))
    centers = F.normalize(prior["centers"].cuda().float(), dim=-1)
    with torch.no_grad():
        z0 = visual_embedding(policy, current)
        z1 = visual_embedding(policy, future)
        delta = F.normalize((z1 - z0).float(), dim=-1)
        labels = (delta @ centers.T).argmax(dim=1)
    action_hidden_dim = int(base.model.action_out_proj.in_features)
    h1_head = VideoDynamicsHead(action_hidden_dim, int(prior["visual_dim"]), centers.shape[0]).cuda()
    bc1, hidden1 = policy_loss_with_action_hidden(policy, current)
    logits, predicted_delta = h1_head(hidden1)
    dyn = F.cross_entropy(logits, labels) + 0.25 * (
        1.0 - F.cosine_similarity(predicted_delta, delta, dim=-1)
    ).mean()
    h1_loss = backward_check("H1", bc1 + 0.10 * dyn, h1_head)

    # H2: instruction-query pooling and a negative seen instruction.
    positive_raw = select_video_time(raw, 0)
    negative_raw = dict(positive_raw)
    processed = target_pre(positive_raw)
    negative_text = _negative_seen_texts(TARGETS[task_id])[0]
    batch_n = int(next(v.shape[0] for v in positive_raw.values() if hasattr(v, "shape") and v.ndim > 0))
    negative_raw["task"] = [negative_text] * batch_n
    negative_processed = target_pre(negative_raw)
    lang_dim = int(base.model.vlm_with_expert.config.text_config.hidden_size)
    action_dim = int(base.config.output_features["action"].shape[0])
    h2_head = InstructionAttentionHead(lang_dim, action_hidden_dim, action_dim).cuda()
    with torch.no_grad():
        q_pos = language_embedding(policy, processed)
        q_neg = language_embedding(policy, negative_processed)
        target_chunk = base.prepare_action(processed)[..., :action_dim]
        action_pad = processed.get("action_is_pad")
        if action_pad is None:
            target_action = target_chunk.mean(dim=1)
        else:
            valid = (~action_pad.bool()).to(target_chunk.dtype).unsqueeze(-1)
            target_action = (target_chunk * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    bc2, hidden2 = policy_loss_with_action_hidden(policy, processed)
    pred_pos = h2_head.predict(hidden2, q_pos)
    pred_neg = h2_head.predict(hidden2, q_neg)
    pos = F.smooth_l1_loss(pred_pos, target_action, reduction="none").mean(dim=1)
    neg = F.smooth_l1_loss(pred_neg, target_action, reduction="none").mean(dim=1)
    h2_aux = pos.mean() + 0.5 * F.relu(0.10 + pos - neg).mean()
    h2_loss = backward_check("H2", bc2 + 0.10 * h2_aux, h2_head)

    # H3: L2-SP path (the penalty is exactly zero before any optimizer step).
    processed3 = target_pre(select_video_time(raw, 0))
    trainable_pairs = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
    reference = {n: p.detach().clone() for n, p in trainable_pairs}
    bc3, _ = policy(processed3)
    squared = torch.stack([(p.float() - reference[n].float()).pow(2).sum() for n, p in trainable_pairs]).sum()
    total_params = sum(p.numel() for _, p in trainable_pairs)
    h3_loss = backward_check("H3", bc3 + 10_000.0 * squared / float(total_params))
    del reference

    # H4: target and seen BC in one shared target-normalized coordinate system.
    seen_ds = temporal_dataset(SEEN_REPO_ID, SEEN_TRAIN_ROOT, cfg)
    seen_pre = make_preprocessor(
        seen_checkpoint,
        policy,
        seen_ds,
        normalization_stats=target_ds.meta.stats,
    )
    seen_loader = make_loader(seen_ds, batch_size, seed + 10_000)
    seen_raw, _ = next_batch(iter(seen_loader), seen_loader)
    bc_target, _ = policy(target_pre(select_video_time(raw, 0)))
    bc_seen, _ = policy(seen_pre(seen_raw))
    h4_loss = backward_check("H4", bc_target + 0.5 * bc_seen)

    result = {
        "task_id": int(task_id),
        "K": int(k),
        "batch_size": int(batch_size),
        "H1_loss": h1_loss,
        "H2_loss": h2_loss,
        "H3_loss": h3_loss,
        "H4_loss": h4_loss,
    }
    del policy, target_loader, target_ds, seen_loader, seen_ds, h1_head, h2_head, raw
    gc.collect()
    torch.cuda.empty_cache()
    return result

def _grid(runner, seen_checkpoint, steps_by_cell, *, batch_size: int, **kwargs):
    out = {}
    for seed in TARGET_SEEDS:
        for task_id in TARGETS:
            for k in BUDGETS:
                out[(task_id, k, seed)] = runner(
                    seen_checkpoint, task_id, k, seed,
                    steps=steps_by_cell[(task_id, k)], batch_size=batch_size, **kwargs,
                )
    return out


def train_h1_grid(seen_checkpoint, prior_path, steps_by_cell, *, batch_size: int, **kwargs):
    out = {}
    for seed in TARGET_SEEDS:
        for task_id in TARGETS:
            for k in BUDGETS:
                out[(task_id, k, seed)] = _run_h1(
                    seen_checkpoint, prior_path, task_id, k, seed,
                    steps=steps_by_cell[(task_id, k)], batch_size=batch_size, **kwargs,
                )
    return out


def train_h2_grid(seen_checkpoint, steps_by_cell, *, batch_size: int, **kwargs):
    return _grid(_run_h2, seen_checkpoint, steps_by_cell, batch_size=batch_size, **kwargs)


def train_h3_grid(seen_checkpoint, steps_by_cell, *, batch_size: int, **kwargs):
    return _grid(_run_h3, seen_checkpoint, steps_by_cell, batch_size=batch_size, **kwargs)


def train_h4_grid(seen_checkpoint, steps_by_cell, *, batch_size: int, **kwargs):
    return _grid(_run_h4, seen_checkpoint, steps_by_cell, batch_size=batch_size, **kwargs)
