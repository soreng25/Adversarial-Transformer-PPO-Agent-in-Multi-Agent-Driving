"""Frozen deterministic adapter around GPUDrive's published PPO policy."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import VictimCheckpointError


def pinned_action_table(environment: dict[str, Any]) -> np.ndarray:
    table = np.asarray(
        [
            [acceleration, steering, head_angle]
            for acceleration, steering, head_angle in itertools.product(
                environment["acceleration_values"],
                environment["steering_values"],
                environment["head_angle_values"],
            )
        ],
        dtype="<f4",
    )
    digest = hashlib.sha256(table.tobytes(order="C")).hexdigest()
    if digest != environment["action_table_float32_sha256"]:
        raise VictimCheckpointError(
            f"pinned action table hash is {digest}, expected "
            f"{environment['action_table_float32_sha256']}"
        )
    return table


def deterministic_argmax(logits: np.ndarray, *, action_dim: int = 91) -> np.ndarray:
    values = np.asarray(logits)
    if values.ndim < 1 or values.shape[-1] != action_dim:
        raise VictimCheckpointError(
            f"expected logits ending in {action_dim}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise VictimCheckpointError("victim logits contain non-finite values")
    return np.argmax(values, axis=-1).astype(np.int64, copy=False)


def validate_slot0_binding(
    controlled_mask: np.ndarray,
    is_sdc: np.ndarray,
    stable_ids: np.ndarray,
    *,
    expected_id: int,
) -> dict[str, Any]:
    controlled = np.asarray(controlled_mask, dtype=bool)
    sdc = np.asarray(is_sdc)
    ids = np.asarray(stable_ids)
    if controlled.shape != sdc.shape or controlled.shape != ids.shape:
        raise VictimCheckpointError("control, SDC, and ID arrays must have equal shape")
    if controlled.ndim != 2 or controlled.shape[0] != 1:
        raise VictimCheckpointError("slot-0 victim binding requires exactly one world")
    if int(controlled.sum()) != 1 or not bool(controlled[0, 0]):
        raise VictimCheckpointError("exactly slot 0 must be controlled")
    if int(sdc[0, 0]) != 1:
        raise VictimCheckpointError("controlled slot 0 is not marked as the SDC")
    if int(ids[0, 0]) != expected_id:
        raise VictimCheckpointError(
            f"slot-0 stable ID is {int(ids[0, 0])}, expected {expected_id}"
        )
    return {"world": 0, "slot": 0, "stable_id": expected_id, "is_sdc": True}


def assert_policy_frozen(policy: Any) -> None:
    if bool(getattr(policy, "training", True)):
        raise VictimCheckpointError("victim policy is not in eval mode")
    parameters = list(policy.parameters())
    if any(bool(parameter.requires_grad) for parameter in parameters):
        raise VictimCheckpointError("victim policy still has trainable parameters")


def torch_module_state_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_frozen_policy(
    checkpoint_directory: Path,
    model_config: dict[str, Any],
    *,
    device: str,
) -> Any:
    try:
        import torch
        from safetensors.torch import load_file
        from gpudrive.networks.late_fusion import NeuralNet
    except Exception as exc:
        raise VictimCheckpointError(
            f"native victim dependencies are unavailable: {type(exc).__name__}: {exc}"
        ) from exc

    policy = NeuralNet(
        action_dim=int(model_config["action_dim"]),
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
        act_func=str(model_config["act_func"]),
        max_controlled_agents=int(model_config["max_controlled_agents"]),
        obs_dim=int(model_config["obs_dim"]),
        config=model_config,
    ).to(device)
    state = load_file(
        str(checkpoint_directory / "model.safetensors"), device=device
    )
    policy.load_state_dict(state, strict=True)
    policy.eval()
    policy.requires_grad_(False)
    assert_policy_frozen(policy)
    return policy


def select_deterministic(policy: Any, observation: Any) -> dict[str, Any]:
    """Run the frozen policy without sampling or autograd."""

    try:
        import torch
    except Exception as exc:
        raise VictimCheckpointError(f"Torch is unavailable: {exc}") from exc
    assert_policy_frozen(policy)
    with torch.inference_mode():
        hidden = policy.encode_observations(observation)
        logits = policy.actor(hidden)
        value = policy.critic(hidden).squeeze(-1)
        if not bool(torch.isfinite(logits).all().item()):
            raise VictimCheckpointError("victim logits contain non-finite values")
        action = torch.argmax(logits, dim=-1)
        log_probability = torch.log_softmax(logits, dim=-1).gather(
            -1, action.unsqueeze(-1)
        ).squeeze(-1)
    return {
        "action": action,
        "logits": logits,
        "value": value,
        "log_probability": log_probability,
    }
