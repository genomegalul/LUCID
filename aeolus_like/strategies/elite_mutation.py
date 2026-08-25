from __future__ import annotations

import jax
import jax.numpy as jnp
from tqdm import tqdm

from config import FitConfig
from parameters import n_raw_params
from strategies.base import SearchResult


def run_elite_mutation(
    config: FitConfig,
    score_batch,
    predict_and_score_batch,
) -> SearchResult:
    """
    GPU-friendly evolutionary strategy.

    Loop:
        1. score population
        2. keep best elites
        3. resample parents from elites
        4. mutate them into the next population
        5. preserve the best elite

    This is not formal MCMC, but it is a strong first optimizer.
    """
    if config.n_elites >= config.n_candidates:
        raise ValueError("n_elites must be smaller than n_candidates.")

    key = jax.random.PRNGKey(config.seed)

    raw_dim = n_raw_params(config)

    key, init_key = jax.random.split(key)
    population = config.raw_init_scale * jax.random.normal(
        init_key,
        shape=(config.n_candidates, raw_dim),
        dtype=jnp.float32,
    )

    best_raw = None
    best_loss = float("inf")
    best_elite_raw = None
    best_elite_losses = None

    loss_history = []

    for step in tqdm(range(config.n_steps), desc="elite_mutation", unit="step"):
        losses = score_batch(population)
        sorted_idx = jnp.argsort(losses)

        elite_idx = sorted_idx[: config.n_elites]
        elites = population[elite_idx]
        elite_losses = losses[elite_idx]

        current_best_loss = float(elite_losses[0])
        current_best_raw = elites[0]

        if current_best_loss < best_loss:
            best_loss = current_best_loss
            best_raw = current_best_raw
            best_elite_raw = elites
            best_elite_losses = elite_losses

        loss_history.append(best_loss)

        key, parent_key, noise_key = jax.random.split(key, 3)

        parent_indices = jax.random.randint(
            parent_key,
            shape=(config.n_candidates,),
            minval=0,
            maxval=config.n_elites,
        )

        parents = elites[parent_indices]

        # Mutation scale decays gently over time.
        frac = step / max(config.n_steps - 1, 1)
        mutation_scale = config.raw_mutation_scale * (1.0 - 0.85 * frac)

        noise = mutation_scale * jax.random.normal(
            noise_key,
            shape=parents.shape,
            dtype=jnp.float32,
        )

        population = parents + noise

        # Preserve current best exactly.
        population = population.at[0].set(current_best_raw)

    best_flux_batch, _ = predict_and_score_batch(best_raw[None, :])
    best_flux = best_flux_batch[0]

    return SearchResult(
        best_raw_params=best_raw,
        best_loss=best_loss,
        elite_raw_params=best_elite_raw,
        elite_losses=best_elite_losses,
        loss_history=jnp.asarray(loss_history, dtype=jnp.float32),
        best_flux=best_flux,
    )