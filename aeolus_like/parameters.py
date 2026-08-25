from __future__ import annotations

from dataclasses import asdict

import jax
import jax.numpy as jnp

from config import FitConfig, ParameterBounds


def n_raw_params(config: FitConfig) -> int:
    """
    Current parameter layout:

    Per spot:
        lat_raw
        lon_raw
        radius_raw
        contrast_raw

    Optional global:
        inclination_raw

    Total:
        4 * n_spots + maybe 1
    """
    n = 4 * config.n_spots
    if config.fit_inclination:
        n += 1
    return n


def sigmoid_to_range(
    x: jnp.ndarray,
    lo: float,
    hi: float,
) -> jnp.ndarray:
    return lo + (hi - lo) * jax.nn.sigmoid(x)


def tanh_to_range(
    x: jnp.ndarray,
    lo: float,
    hi: float,
) -> jnp.ndarray:
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return mid + half * jnp.tanh(x)


def unpack_physical_params(
    raw_params: jnp.ndarray,
    config: FitConfig,
) -> dict:
    """
    Convert unconstrained raw parameters to physical spot parameters.

    raw_params shape:
        (n_raw_params,)

    Returns:
        {
            lats:        (n_spots,)
            lons:        (n_spots,)
            radii:       (n_spots,)
            contrasts:   (n_spots,)
            inclination: scalar
        }
    """
    bounds: ParameterBounds = config.bounds

    spot_raw = raw_params[: 4 * config.n_spots].reshape((config.n_spots, 4))

    # Mild scaling prevents tanh/sigmoid from saturating too early.
    scale = 0.3

    lats = tanh_to_range(
        scale * spot_raw[:, 0],
        bounds.lat_min_deg,
        bounds.lat_max_deg,
    )

    lons = tanh_to_range(
        scale * spot_raw[:, 1],
        bounds.lon_min_deg,
        bounds.lon_max_deg,
    )

    radii = sigmoid_to_range(
        scale * spot_raw[:, 2],
        bounds.radius_min_deg,
        bounds.radius_max_deg,
    )

    contrasts = sigmoid_to_range(
        scale * spot_raw[:, 3],
        bounds.contrast_min,
        bounds.contrast_max,
    )

    if config.fit_inclination:
        inc_raw = raw_params[-1]
        inclination = sigmoid_to_range(
            scale * inc_raw,
            bounds.inclination_min_deg,
            bounds.inclination_max_deg,
        )
    else:
        if config.fixed_inclination_deg is None:
            raise ValueError(
                "fixed_inclination_deg must be provided when fit_inclination=False"
            )
        inclination = jnp.asarray(config.fixed_inclination_deg, dtype=jnp.float32)

    return {
        "lats": lats.astype(jnp.float32),
        "lons": lons.astype(jnp.float32),
        "radii": radii.astype(jnp.float32),
        "contrasts": contrasts.astype(jnp.float32),
        "inclination": inclination.astype(jnp.float32),
    }


def angular_distance_deg(
    lat1: jnp.ndarray,
    lon1: jnp.ndarray,
    lat2: jnp.ndarray,
    lon2: jnp.ndarray,
) -> jnp.ndarray:
    lat1 = jnp.deg2rad(lat1)
    lon1 = jnp.deg2rad(lon1)
    lat2 = jnp.deg2rad(lat2)
    lon2 = jnp.deg2rad(lon2)

    cos_sep = (
        jnp.sin(lat1) * jnp.sin(lat2)
        + jnp.cos(lat1) * jnp.cos(lat2) * jnp.cos(lon1 - lon2)
    )

    cos_sep = jnp.clip(cos_sep, -1.0, 1.0)
    return jnp.rad2deg(jnp.arccos(cos_sep))


def spot_separation_penalty(
    raw_params: jnp.ndarray,
    config: FitConfig,
) -> jnp.ndarray:
    """
    Penalize overlapping or too-close spots.

    This is not required, but it helps enforce the same physical structure
    you introduced in the synthesis data.
    """
    if config.n_spots <= 1:
        return jnp.asarray(0.0, dtype=jnp.float32)

    phys = unpack_physical_params(raw_params, config)

    lats = phys["lats"]
    lons = phys["lons"]
    radii = phys["radii"]

    penalty = jnp.asarray(0.0, dtype=jnp.float32)

    for i in range(config.n_spots):
        for j in range(i + 1, config.n_spots):
            sep = angular_distance_deg(lats[i], lons[i], lats[j], lons[j])
            min_allowed = radii[i] + radii[j] + config.min_spot_gap_deg
            violation = jnp.maximum(0.0, min_allowed - sep)
            penalty = penalty + violation**2

    return penalty.astype(jnp.float32)


def physical_params_to_numpy_dict(
    raw_params,
    config: FitConfig,
) -> dict:
    """
    Convenience helper for saving best-fit physical parameters.

    This function should be called outside JIT.
    """
    phys = unpack_physical_params(jnp.asarray(raw_params, dtype=jnp.float32), config)
    return {k: jnp.asarray(v).block_until_ready() for k, v in phys.items()}


def config_to_jsonable(config: FitConfig) -> dict:
    d = asdict(config)
    d["output_dir"] = str(config.output_dir)
    return d