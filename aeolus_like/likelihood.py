from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit

from config import FitConfig
from parameters import spot_separation_penalty


def make_objective(
    config: FitConfig,
    simulate_batch,
    target_flux,
):
    """
    Build batched objective.

    Returns:
        score_batch(raw_params_batch) -> losses

    Lower is better.
    """
    target_flux = jnp.asarray(target_flux, dtype=jnp.float32)

    def mse_single(pred_flux: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean((pred_flux - target_flux) ** 2)

    def loss_single(raw_params: jnp.ndarray, pred_flux: jnp.ndarray) -> jnp.ndarray:
        mse = mse_single(pred_flux)

        if config.use_separation_penalty:
            sep_penalty = spot_separation_penalty(raw_params, config)
            return mse + config.separation_penalty_weight * sep_penalty

        return mse

    @jit
    def score_batch(raw_params_batch: jnp.ndarray) -> jnp.ndarray:
        pred_batch = simulate_batch(raw_params_batch)

        losses = jax.vmap(loss_single)(
            raw_params_batch,
            pred_batch,
        )

        return losses.astype(jnp.float32)

    @jit
    def predict_and_score_batch(raw_params_batch: jnp.ndarray):
        pred_batch = simulate_batch(raw_params_batch)

        losses = jax.vmap(loss_single)(
            raw_params_batch,
            pred_batch,
        )

        return pred_batch, losses.astype(jnp.float32)

    return score_batch, predict_and_score_batch