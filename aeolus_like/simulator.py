from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

import jax
import jax.numpy as jnp
from jax import jit

from jaxoplanet.starry.ylm import Ylm, ylm_spot
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.light_curves import surface_light_curve

from config import FitConfig


YDEG = 11
YSIZE = (YDEG + 1) * (YDEG + 1)


def initialize_spot_function():
    """
    ylm_spot initialization can trigger GPU linear algebra issues on some systems.
    Initialize it on CPU, then use it inside the JAX graph.
    """
    print("[simulator] initializing ylm_spot on CPU...", flush=True)

    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        spot_fn = ylm_spot(YDEG)

    print("[simulator] ylm_spot initialized.", flush=True)
    return spot_fn


SPOT_FN = initialize_spot_function()
UNIT_DENSE = jnp.zeros(YSIZE, dtype=jnp.float32).at[0].set(1.0)


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


def unpack_spot_params_only(
    raw_params: jnp.ndarray,
    config: FitConfig,
) -> dict:
    """
    Convert raw parameters to physical spot parameters.

    This intentionally does NOT rely on config.fixed_inclination_deg.
    Inclination is handled dynamically in make_simulator() so that changing
    inclination from example to example does not trigger recompilation.

    Parameter layout:

    If config.fit_inclination is False:
        raw_params = [
            lat_1, lon_1, radius_1, contrast_1,
            ...
            lat_N, lon_N, radius_N, contrast_N
        ]

    If config.fit_inclination is True:
        raw_params = [
            lat_1, lon_1, radius_1, contrast_1,
            ...
            lat_N, lon_N, radius_N, contrast_N,
            inclination
        ]
    """
    bounds = config.bounds
    scale = 0.3

    spot_raw = raw_params[: 4 * config.n_spots].reshape((config.n_spots, 4))

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

    return {
        "lats": lats.astype(jnp.float32),
        "lons": lons.astype(jnp.float32),
        "radii": radii.astype(jnp.float32),
        "contrasts": contrasts.astype(jnp.float32),
    }


def dynamic_inclination_from_raw_or_input(
    raw_params: jnp.ndarray,
    inclination_deg: jnp.ndarray,
    config: FitConfig,
) -> jnp.ndarray:
    """
    Use the raw inclination parameter if config.fit_inclination=True.
    Otherwise, use the runtime inclination_deg argument.

    This keeps inclination dynamic for --use-true-inclination and --inclination,
    avoiding recompilation when inclination changes between examples.
    """
    bounds = config.bounds
    scale = 0.3

    if config.fit_inclination:
        inc_raw = raw_params[-1]
        inc = sigmoid_to_range(
            scale * inc_raw,
            bounds.inclination_min_deg,
            bounds.inclination_max_deg,
        )
        return inc.astype(jnp.float32)

    return jnp.asarray(inclination_deg, dtype=jnp.float32)


def build_map_from_spot_params(
    spot_params: dict,
    config: FitConfig,
) -> jnp.ndarray:
    lats = spot_params["lats"]
    lons = spot_params["lons"]
    radii = spot_params["radii"]
    contrasts = spot_params["contrasts"]

    def add_one(i, y_curr):
        spot = SPOT_FN(
            contrast=contrasts[i],
            r=jnp.deg2rad(radii[i]),
            lat=jnp.deg2rad(lats[i]),
            lon=jnp.deg2rad(lons[i]),
        )

        y_spot = spot.todense()
        return y_curr + (y_spot - UNIT_DENSE)

    return jax.lax.fori_loop(
        0,
        config.n_spots,
        add_one,
        UNIT_DENSE,
    )


def make_simulator(
    config: FitConfig,
    theta_deg,
):
    """
    Create a jitted batched simulator for one fixed config and theta grid.

    Important:
        inclination_deg is a runtime argument, not closed over in config.

    Returns:
        simulate_batch(raw_params_batch, inclination_deg) -> flux_batch

    Shapes:
        raw_params_batch: (n_candidates, n_params)
        inclination_deg: scalar
        output:          (n_candidates, datapoints)

    If config.fit_inclination=True, inclination_deg is ignored and inclination
    is taken from raw_params.
    """
    theta_deg = jnp.asarray(theta_deg, dtype=jnp.float32)

    @jit
    def flux_from_map_ylm(
        y_dense: jnp.ndarray,
        inc_deg: jnp.ndarray,
    ) -> jnp.ndarray:
        surface = Surface(
            y=Ylm.from_dense(y_dense, normalize=False),
            inc=jnp.deg2rad(inc_deg),
        )

        def step(theta):
            return surface_light_curve(surface, theta=jnp.deg2rad(theta))

        flux = jax.vmap(step)(theta_deg)

        return jnp.nan_to_num(
            flux.astype(jnp.float32),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )

    def simulate_single(
        raw_params: jnp.ndarray,
        inclination_deg: jnp.ndarray,
    ) -> jnp.ndarray:
        spot_params = unpack_spot_params_only(raw_params, config)

        inc = dynamic_inclination_from_raw_or_input(
            raw_params=raw_params,
            inclination_deg=inclination_deg,
            config=config,
        )

        y_dense = build_map_from_spot_params(
            spot_params=spot_params,
            config=config,
        )

        return flux_from_map_ylm(
            y_dense=y_dense,
            inc_deg=inc,
        )

    @jit
    def simulate_batch(
        raw_params_batch: jnp.ndarray,
        inclination_deg: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(
            lambda raw: simulate_single(raw, inclination_deg)
        )(raw_params_batch)

    return simulate_batch