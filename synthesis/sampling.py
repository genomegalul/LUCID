from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
from jax import jit


# --- Parameter Ranges ---

DATAPOINTS = 128
# When varying datapoints at the cost of recompilations:
# DATAPOINTS_RANGE = (32, 512)
NOISE_FRACTION_RANGE = (0.0, 0.10)
INCLINATION_DEG_RANGE = (10.0, 90.0)

SPOT_LAT_DEG_RANGE = (-90.0, 90.0)
SPOT_LON_DEG_RANGE = (-180.0, 180.0)
SPOT_RADIUS_DEG_RANGE = (4.0, 18.0)


# --- Sampling Utilities ---

@jit
def u01(rng_key):
    return jax.random.uniform(rng_key, dtype=jnp.float32)


@jit
def uniform(rng_key, lo, hi):
    lo = jnp.asarray(lo, dtype=jnp.float32)
    hi = jnp.asarray(hi, dtype=jnp.float32)
    return lo + (hi - lo) * u01(rng_key)


@jit
def randint_inclusive(rng_key, lo, hi):
    return jax.random.randint(rng_key, shape=(), minval=lo, maxval=hi + 1)


# --- Sampling Function ---

def sample_parameters(rng_key, n_spots: int) -> Dict[str, jnp.ndarray]:
    """
    Sample one set of observation and spot parameters.

    Returns:
      - datapoints
      - noise
      - inclination
      - n_spots
      - lats
      - lons
      - radii
    """
    
    n_parameter_subkeys = 4 + 3 * n_spots
    parameter_keys = jax.random.split(rng_key, n_parameter_subkeys)

    k_datapoints = parameter_keys[0]
    k_noise = parameter_keys[1]
    k_inclination = parameter_keys[2]
    k_lats = parameter_keys[3 : 3 + n_spots]
    k_lons = parameter_keys[3 + n_spots : 3 + 2 * n_spots]
    k_radii = parameter_keys[3 + 2 * n_spots : 3 + 3 * n_spots]

    datapoints = jnp.asarray(DATAPOINTS, dtype=jnp.int32)
    # When varying datapoints at the cost of recompilations:
    # datapoints = randint_inclusive(k_datapoints, *DATAPOINTS_RANGE)
    noise = uniform(k_noise, *NOISE_FRACTION_RANGE)
    inclination = uniform(k_inclination, *INCLINATION_DEG_RANGE)

    lats = jax.vmap(lambda k: uniform(k, *SPOT_LAT_DEG_RANGE))(k_lats)
    lons = jax.vmap(lambda k: uniform(k, *SPOT_LON_DEG_RANGE))(k_lons)
    radii = jax.vmap(lambda k: uniform(k, *SPOT_RADIUS_DEG_RANGE))(k_radii)

    sampled_parameters = {
        "datapoints": jnp.asarray(datapoints, dtype=jnp.int32),
        "noise": jnp.asarray(noise, dtype=jnp.float32),
        "inclination": jnp.asarray(inclination, dtype=jnp.float32),
        "n_spots": jnp.asarray(n_spots, dtype=jnp.int32),
        "lats": jnp.asarray(lats, dtype=jnp.float32),
        "lons": jnp.asarray(lons, dtype=jnp.float32),
        "radii": jnp.asarray(radii, dtype=jnp.float32),
    }

    return sampled_parameters