from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class SearchResult:
    best_raw_params: jnp.ndarray
    best_loss: float

    elite_raw_params: jnp.ndarray
    elite_losses: jnp.ndarray

    loss_history: jnp.ndarray

    best_flux: jnp.ndarray | None = None