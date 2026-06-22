#!/usr/bin/env python3
"""
Lightweight Titans Long-Term Memory module for the mag_demo project.

Goal
----
This file isolates the memory idea from Titans / Aedelon/titans-pytorch-mlx
without replacing the whole LLM architecture.

It implements only the Neural Long-Term Memory behavior needed by our
external-memory agent:
- key/value/query projections;
- associative memory loss M(k) -> v;
- test-time update through gradient descent;
- surprise momentum;
- weight decay / forgetting;
- retrieve-only mode.

Expected tensor shape: (batch, sequence, dim).
The module is intentionally small and CPU-friendly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


ActivationName = Literal["silu", "gelu", "relu"]


@dataclass
class TitansMemoryConfig:
    """Configuration for the standalone Titans long-term memory module."""

    dim: int = 128
    num_memory_layers: int = 2
    memory_hidden_mult: float = 2.0
    memory_lr: float = 0.05
    memory_momentum: float = 0.90
    memory_decay: float = 0.001
    activation: ActivationName = "gelu"
    init_std: float = 0.02

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_memory_layers < 1:
            raise ValueError("num_memory_layers must be >= 1")
        if not 0.0 < self.memory_lr <= 1.0:
            raise ValueError("memory_lr must be in (0, 1]")
        if not 0.0 <= self.memory_momentum < 1.0:
            raise ValueError("memory_momentum must be in [0, 1)")
        if not 0.0 <= self.memory_decay < 1.0:
            raise ValueError("memory_decay must be in [0, 1)")

    @property
    def memory_hidden_dim(self) -> int:
        return max(self.dim, int(self.dim * self.memory_hidden_mult))


def get_activation(name: ActivationName) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation: {name}")


@dataclass
class MemoryState:
    """External state of the neural memory.

    weights represent M_t in the Titans paper.
    momentum represents S_t, the accumulated surprise.
    """

    weights: list[torch.Tensor]
    momentum: list[torch.Tensor]

    def detach(self) -> "MemoryState":
        return MemoryState(
            weights=[w.detach().clone() for w in self.weights],
            momentum=[m.detach().clone() for m in self.momentum],
        )

    def clone(self) -> "MemoryState":
        return MemoryState(
            weights=[w.clone() for w in self.weights],
            momentum=[m.clone() for m in self.momentum],
        )


class MemoryMLP(nn.Module):
    """The neural memory M.

    A 1-layer memory is equivalent to a linear associative memory.
    A 2+ layer memory is closer to the deep memory module discussed in Titans.
    """

    def __init__(self, config: TitansMemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList()

        if config.num_memory_layers == 1:
            self.layers.append(nn.Linear(config.dim, config.dim, bias=False))
        else:
            self.layers.append(nn.Linear(config.dim, config.memory_hidden_dim, bias=False))
            for _ in range(config.num_memory_layers - 2):
                self.layers.append(
                    nn.Linear(config.memory_hidden_dim, config.memory_hidden_dim, bias=False)
                )
            self.layers.append(nn.Linear(config.memory_hidden_dim, config.dim, bias=False))

        self.activation = get_activation(config.activation)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in self.layers:
            nn.init.normal_(layer.weight, mean=0.0, std=self.config.init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for idx, layer in enumerate(self.layers):
            h = layer(h)
            if idx < len(self.layers) - 1:
                h = self.activation(h)
        return h

    def get_weights(self) -> list[torch.Tensor]:
        return [layer.weight.detach().clone() for layer in self.layers]

    def set_weights(self, weights: list[torch.Tensor]) -> None:
        if len(weights) != len(self.layers):
            raise ValueError(
                f"Expected {len(self.layers)} weight tensors, got {len(weights)}"
            )
        with torch.no_grad():
            for layer, weight in zip(self.layers, weights, strict=True):
                layer.weight.copy_(weight.to(layer.weight.device, dtype=layer.weight.dtype))

    def associative_loss(self, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self.forward(keys), values, reduction="mean")


class NeuralLongTermMemory(nn.Module):
    """Standalone Titans-style Neural Long-Term Memory.

    It performs retrieval using the current state, then optionally updates the
    memory weights using the associative loss:

        loss = || M(k_t) - v_t ||²

    The update follows the Titans intuition:

        S_t = eta * S_{t-1} - theta * grad(loss)
        M_t = (1 - alpha) * M_{t-1} + S_t

    where alpha is forgetting/decay, eta is surprise momentum, and theta is the
    test-time memory learning rate.
    """

    def __init__(self, config: TitansMemoryConfig | None = None) -> None:
        super().__init__()
        self.config = config or TitansMemoryConfig()
        dim = self.config.dim

        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, dim, bias=False)
        self.memory = MemoryMLP(self.config)

        self.reset_parameters()

        # The projections are part of the memory interface, not trained at test time here.
        for module in (self.proj_k, self.proj_v, self.proj_q, self.proj_out):
            for param in module.parameters():
                param.requires_grad_(False)

    def reset_parameters(self) -> None:
        for module in (self.proj_k, self.proj_v, self.proj_q, self.proj_out):
            nn.init.eye_(module.weight)

    def init_state(self, device: torch.device | str = "cpu") -> MemoryState:
        device = torch.device(device)
        weights = [w.detach().clone().to(device) for w in self.memory.get_weights()]
        momentum = [torch.zeros_like(w, device=device) for w in weights]
        return MemoryState(weights=weights, momentum=momentum)

    def project_keys_values_queries(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keys = F.normalize(F.silu(self.proj_k(x)), p=2, dim=-1)
        values = F.normalize(F.silu(self.proj_v(x)), p=2, dim=-1)
        queries = F.normalize(F.silu(self.proj_q(x)), p=2, dim=-1)
        return keys, values, queries

    def retrieve(self, queries: torch.Tensor, state: MemoryState) -> torch.Tensor:
        self.memory.set_weights(state.weights)
        _, _, projected_queries = self.project_keys_values_queries(queries)
        retrieved = self.memory(projected_queries)
        return self.proj_out(retrieved)

    def forward(
        self,
        x: torch.Tensor,
        state: MemoryState | None = None,
        update: bool = True,
    ) -> tuple[torch.Tensor, MemoryState]:
        if x.ndim != 3:
            raise ValueError("x must have shape (batch, sequence, dim)")
        if x.shape[-1] != self.config.dim:
            raise ValueError(f"Expected last dim {self.config.dim}, got {x.shape[-1]}")

        if state is None:
            state = self.init_state(x.device)

        self.memory.set_weights(state.weights)
        keys, values, queries = self.project_keys_values_queries(x)

        # Read before write: M*_{t-1}(q_t)
        retrieved = self.memory(queries)
        output = self.proj_out(retrieved)

        if not update:
            return output, state.detach()

        grads = self._compute_gradients(keys, values)
        new_state = self._update_state(state, grads)
        return output, new_state.detach()

    def _compute_gradients(
        self, keys: torch.Tensor, values: torch.Tensor
    ) -> list[torch.Tensor]:
        for param in self.memory.parameters():
            param.requires_grad_(True)

        loss = self.memory.associative_loss(keys.detach(), values.detach())
        grads = torch.autograd.grad(
            loss,
            list(self.memory.parameters()),
            create_graph=False,
            allow_unused=True,
        )

        out: list[torch.Tensor] = []
        for grad, param in zip(grads, self.memory.parameters(), strict=True):
            out.append(torch.zeros_like(param) if grad is None else grad.detach())
            param.requires_grad_(False)
        return out

    def _update_state(self, state: MemoryState, grads: list[torch.Tensor]) -> MemoryState:
        alpha = self.config.memory_decay
        eta = self.config.memory_momentum
        theta = self.config.memory_lr

        new_weights: list[torch.Tensor] = []
        new_momentum: list[torch.Tensor] = []

        for weight, momentum, grad in zip(
            state.weights, state.momentum, grads, strict=True
        ):
            grad = grad.to(weight.device, dtype=weight.dtype)
            surprise = eta * momentum - theta * grad
            updated_weight = (1.0 - alpha) * weight + surprise
            new_momentum.append(surprise)
            new_weights.append(updated_weight)

        return MemoryState(weights=new_weights, momentum=new_momentum)

    def associative_loss_for_input(
        self, x: torch.Tensor, state: MemoryState | None = None
    ) -> float:
        if state is None:
            state = self.init_state(x.device)
        self.memory.set_weights(state.weights)
        keys, values, _ = self.project_keys_values_queries(x)
        with torch.no_grad():
            loss = self.memory.associative_loss(keys, values)
        return float(loss.item())

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
