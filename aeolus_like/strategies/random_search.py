from __future__ import annotations

import jax
import jax.numpy as jnp

from config import FitConfig
from parameters import n_raw_params
from strategies.base import SearchResult


def run_random_search(
    config: FitConfig,
    score_batch,
    predict_and_score_batch,
) -> SearchResult:
    """
    Simple GPU-batched random search.

    This is mainly a sanity check strategy:
        sample many candidate maps,
        score all,
        keep best.
    """
    key = jax.random.PRNGKey(config.seed)

    raw_dim = n_raw_params(config)

    candidates = config.raw_init_scale * jax.random.normal(
        key,
        shape=(config.n_candidates, raw_dim),
        dtype=jnp.float32,
    )

    losses = score_batch(candidates)
    sorted_idx = jnp.argsort(losses)

    elite_idx = sorted_idx[: config.n_elites]

    elite_raw = candidates[elite_idx]
    elite_losses = losses[elite_idx]

    best_raw = elite_raw[0]
    best_loss = float(elite_losses[0])

    best_flux_batch, _ = predict_and_score_batch(best_raw[None, :])
    best_flux = best_flux_batch[0]

    return SearchResult(
        best_raw_params=best_raw,
        best_loss=best_loss,
        elite_raw_params=elite_raw,
        elite_losses=elite_losses,
        loss_history=jnp.asarray([best_loss], dtype=jnp.float32),
        best_flux=best_flux,
    )