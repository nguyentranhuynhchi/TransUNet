"""AdamW optimizer and polynomial learning rate scheduler matching SCAR / I-MMSeg benchmark."""
from __future__ import annotations

from typing import Any
import torch

OPTIMIZATION_PROTOCOL = {
    "id": "im_mseg_adamw_poly_v1",
    "optimizer": "AdamW",
    "weight_decay_scope": "all_parameters",
    "betas": [0.9, 0.999],
    "eps": 1e-8,
    "scheduler": "polynomial_after_update_zero_based_iteration",
    "power": 0.9,
}


def build_optimizer(model: torch.nn.Module, base_lr: float = 0.0003333, weight_decay: float = 0.0001) -> torch.optim.Optimizer:
    """Build AdamW optimizer applied to all 105.17M model parameters."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, total_updates: int = 37200) -> torch.optim.lr_scheduler.LambdaLR:
    """Build polynomial decay scheduler with power 0.9 and zero-based offset.
    
    Formula: lr_u = base_lr * max(0, 1 - max(0, u - 1) / total_updates) ** 0.9
    First two updates maintain base_lr; step 37200 has lr > 0.
    """
    if not isinstance(total_updates, int) or total_updates < 1:
        raise ValueError("total_updates must be a positive integer")

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(0.0, 1.0 - max(0, step - 1) / total_updates) ** 0.9,
    )


def validate_optimization_checkpoint(checkpoint: dict[str, Any]) -> None:
    """Validate that resumed checkpoint uses the same optimization protocol."""
    if checkpoint.get("optimization_protocol") != OPTIMIZATION_PROTOCOL:
        raise ValueError(
            "Resume optimizer/LR scheduler protocol differs from benchmark. "
            f"Expected {OPTIMIZATION_PROTOCOL}, found {checkpoint.get('optimization_protocol')}"
        )
