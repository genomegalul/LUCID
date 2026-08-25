from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp
from jax import jit


# --- Parameter Ranges ---

# Per map
DATAPOINTS = 256
NOISE_FRACTION_RANGE = (0.0, 0.10)
INCLINATION_DEG_RANGE = (10.0, 90.0)
SPOT_CONTRAST_RANGE = (0.3, 0.8)

# Per spot
SPOT_RADIUS_DEG_RANGE = (4.0, 18.0)
SPOT_LAT_DEG_RANGE = (-90.0, 90.0)
SPOT_LON_DEG_RANGE = (-180.0, 180.0)

# Sampling parameters
MIN_SPOT_GAP_DEG = 2.0
MAX_SPOT_PLACEMENT_TRIES = 500


# --- Sampling Utilities ---

@jit
def u01(rng_key):
    return jax.random.uniform(rng_key, dtype=jnp.float32)


@jit
def uniform(rng_key, lo, hi):
    lo = jnp.asarray(lo, dtype=jnp.float32)
    hi = jnp.asarray(hi, dtype=jnp.float32)
    return lo + (hi - lo) * u01(rng_key)


def angular_distance_deg(lat1, lon1, lat2, lon2):
    """
    Angular separation in degrees between two spherical points.
    """
    
    lat1 = jnp.deg2rad(jnp.asarray(lat1, dtype=jnp.float32))
    lon1 = jnp.deg2rad(jnp.asarray(lon1, dtype=jnp.float32))
    lat2 = jnp.deg2rad(jnp.asarray(lat2, dtype=jnp.float32))
    lon2 = jnp.deg2rad(jnp.asarray(lon2, dtype=jnp.float32))

    cos_sep = (
        jnp.sin(lat1) * jnp.sin(lat2)
        + jnp.cos(lat1) * jnp.cos(lat2) * jnp.cos(lon1 - lon2)
    )
    cos_sep = jnp.clip(cos_sep, -1.0, 1.0)
    return jnp.rad2deg(jnp.arccos(cos_sep))


def spots_are_separated(
    cand_lat,
    cand_lon,
    cand_radius,
    existing_lats,
    existing_lons,
    existing_radii,
    min_gap_deg,
):
    """
    Return True if the candidate spot does not overlap any existing spot.
    """
    
    for lat, lon, radius in zip(existing_lats, existing_lons, existing_radii):
        sep = angular_distance_deg(cand_lat, cand_lon, lat, lon)
        min_allowed = cand_radius + radius + min_gap_deg
        if sep < min_allowed:
            return False
    return True


# --- Sampling Function ---

def sample_parameters(rng_key, n_spots: int) -> Dict[str, jnp.ndarray]:
    """
    Sample one set of observation and spot parameters.

    Returns:
      - datapoints
      - noise
      - inclination
      - contrast
      - n_spots
      - radii
      - lats
      - lons

    Spots are placed sequentially with rejection sampling so they do not overlap.
    """
    k_noise, k_inclination, k_contrast, k_spots = jax.random.split(rng_key, 4)

    datapoints = jnp.asarray(DATAPOINTS, dtype=jnp.int32)
    noise = uniform(k_noise, *NOISE_FRACTION_RANGE)
    inclination = uniform(k_inclination, *INCLINATION_DEG_RANGE)
    contrast = uniform(k_contrast, *SPOT_CONTRAST_RANGE)

    placed_radii = []
    placed_lats = []
    placed_lons = []

    spot_key = k_spots

    for _ in range(n_spots):
        accepted = False

        for _try in range(MAX_SPOT_PLACEMENT_TRIES):
            spot_key, k_r, k_lat, k_lon = jax.random.split(spot_key, 4)

            cand_radius = uniform(k_r, *SPOT_RADIUS_DEG_RANGE)
            cand_lat = uniform(k_lat, *SPOT_LAT_DEG_RANGE)
            cand_lon = uniform(k_lon, *SPOT_LON_DEG_RANGE)

            if spots_are_separated(
                cand_lat=cand_lat,
                cand_lon=cand_lon,
                cand_radius=cand_radius,
                existing_lats=placed_lats,
                existing_lons=placed_lons,
                existing_radii=placed_radii,
                min_gap_deg=MIN_SPOT_GAP_DEG,
            ):
                placed_radii.append(cand_radius)
                placed_lats.append(cand_lat)
                placed_lons.append(cand_lon)
                accepted = True
                break

        if not accepted:
            raise RuntimeError(
                f"Could not place {n_spots} non-overlapping spots after "
                f"{MAX_SPOT_PLACEMENT_TRIES} tries per spot. "
                f"Try reducing spot radii, reducing n_spots, or reducing MIN_SPOT_GAP_DEG."
            )

    radii = jnp.asarray(placed_radii, dtype=jnp.float32)
    lats = jnp.asarray(placed_lats, dtype=jnp.float32)
    lons = jnp.asarray(placed_lons, dtype=jnp.float32)

    sampled_parameters = {
        "datapoints": jnp.asarray(datapoints, dtype=jnp.int32),
        "noise": jnp.asarray(noise, dtype=jnp.float32),
        "inclination": jnp.asarray(inclination, dtype=jnp.float32),
        "contrast": jnp.asarray(contrast, dtype=jnp.float32),
        "n_spots": jnp.asarray(n_spots, dtype=jnp.int32),
        "radii": radii,
        "lats": lats,
        "lons": lons,
    }

    return sampled_parameters


def sample_parameters_batch(keys, n_spots: int) -> Dict[str, jnp.ndarray]:
    """
    Sample a batch of observation and spot parameter sets from per-example keys.
    """
    
    samples = [sample_parameters(k, n_spots) for k in list(keys)]

    return {
        "datapoints": jnp.stack([s["datapoints"] for s in samples]),
        "noise": jnp.stack([s["noise"] for s in samples]),
        "inclination": jnp.stack([s["inclination"] for s in samples]),
        "contrast": jnp.stack([s["contrast"] for s in samples]),
        "n_spots": jnp.stack([s["n_spots"] for s in samples]),
        "radii": jnp.stack([s["radii"] for s in samples]),
        "lats": jnp.stack([s["lats"] for s in samples]),
        "lons": jnp.stack([s["lons"] for s in samples]),
    }