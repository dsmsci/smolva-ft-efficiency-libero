from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class DynamicsConfig:
    latent_dim: int
    horizon: int = 10
    action_dim: int = 7
    hidden_dim: int = 256


class LatentDynamicsHead(nn.Module):
    def __init__(self, config: DynamicsConfig):
        super().__init__()
        self.config = config
        self.gru = nn.GRU(
            config.action_dim,
            config.hidden_dim,
            batch_first=True,
        )
        self.predictor = nn.Sequential(
            nn.Linear(config.latent_dim + config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )

    def forward(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(actions)
        delta = self.predictor(torch.cat([z, hidden[-1]], dim=-1))
        return z + delta


def unwrap_policy(policy):
    return policy.get_base_model() if hasattr(policy, "peft_config") else policy


def prefix_latent(policy, batch: dict) -> torch.Tensor:
    base = unwrap_policy(policy)
    images, image_masks = base.prepare_images(batch)
    state = base.prepare_state(batch)
    tokens = batch["observation.language.tokens"]
    token_mask = batch["observation.language.attention_mask"].bool()
    prefix, pad_mask, _ = base.model.embed_prefix(
        images,
        image_masks,
        tokens,
        token_mask,
        state=state,
    )
    mask = pad_mask.to(prefix.dtype).unsqueeze(-1)
    return (prefix * mask).sum(1) / mask.sum(1).clamp_min(1)


def dynamics_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
    contrastive_weight: float = 0.1,
    temperature: float = 0.1,
) -> torch.Tensor:
    target = target.detach()
    if valid is not None:
        prediction, target = prediction[valid], target[valid]
    if prediction.shape[0] == 0:
        return prediction.sum() * 0

    loss = F.smooth_l1_loss(prediction, target)
    if prediction.shape[0] < 2:
        return loss

    logits = (
        F.normalize(prediction, dim=-1)
        @ F.normalize(target, dim=-1).T
        / temperature
    )
    labels = torch.arange(prediction.shape[0], device=prediction.device)
    return loss + contrastive_weight * F.cross_entropy(logits, labels)


def flow_clean_action(policy, batch: dict, noise: torch.Tensor, time: torch.Tensor):
    base = unwrap_policy(policy)
    captured = {}

    def capture_velocity(_module, _inputs, output):
        captured["velocity"] = output

    handle = base.model.action_out_proj.register_forward_hook(capture_velocity)
    try:
        per_sample_loss, _ = policy(
            batch,
            noise=noise,
            time=time,
            reduction="none",
        )
    finally:
        handle.remove()

    if "velocity" not in captured:
        raise RuntimeError("SmolVLA action_out_proj did not return flow velocity")

    target = base.prepare_action(batch)
    velocity = captured["velocity"]
    if velocity.shape != target.shape:
        raise RuntimeError(
            f"Velocity shape {tuple(velocity.shape)} != action chunk {tuple(target.shape)}"
        )

    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * target
    clean_action = x_t - time[:, None, None] * velocity
    action_dim = base.config.output_features["action"].shape[0]
    return per_sample_loss, clean_action[..., :action_dim]
