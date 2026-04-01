from __future__ import annotations

import os
from typing import Dict

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

import jax
import jax.numpy as jnp
from jax import jit

from jaxoplanet.starry.ylm import Ylm, ylm_spot
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.light_curves import surface_light_curve


# --- Simulator Constants ---

YDEG = 11
YSIZE = (YDEG + 1) * (YDEG + 1)

FIXED_SPOT_CONTRAST = 0.55

UNIT_DENSE = jnp.zeros(YSIZE, dtype=jnp.float32).at[0].set(1.0)


# --- jaxoplanet Setup ---

print("Forcing 'ylm_spot' initialization to CPU to bypass potential cuSOLVER issues...")
try:
    cpu_device = jax.devices("cpu")[0]
except IndexError as exc:
    raise RuntimeError("No CPU device found by JAX. This workaround requires a CPU.") from exc

with jax.default_device(cpu_device):
    SPOT_FN = ylm_spot(YDEG)

print("... 'ylm_spot' initialized successfully on CPU.")


# --- Simulator Utilities ---

def make_theta_grid(datapoints: int):
    return jnp.linspace(0.0, 360.0, datapoints, endpoint=False, dtype=jnp.float32)


@jit
def build_spot_map(lats: jnp.ndarray, lons: jnp.ndarray, radii: jnp.ndarray) -> jnp.ndarray:
    """
    Build a dense Y_lm map from sampled spot parameters.
    """
    n_spots = lats.shape[0]

    def add_one(i, y_curr):
        spot = SPOT_FN(
            contrast=FIXED_SPOT_CONTRAST,
            r=jnp.deg2rad(radii[i]),
            lat=jnp.deg2rad(lats[i]),
            lon=jnp.deg2rad(lons[i]),
        )
        y_spot = spot.todense()
        return y_curr + (y_spot - UNIT_DENSE)

    return jax.lax.fori_loop(0, n_spots, add_one, UNIT_DENSE)


@jit
def flux_from_map_ylm(y_dense: jnp.ndarray, inclination: float, theta_deg: jnp.ndarray) -> jnp.ndarray:
    surface = Surface(
        y=Ylm.from_dense(y_dense, normalize=False),
        inc=jnp.deg2rad(inclination),
    )

    def step(theta):
        return surface_light_curve(surface, theta=jnp.deg2rad(theta))

    return jax.vmap(step)(theta_deg).astype(jnp.float32)


@jit
def add_flux_noise(rng_key, flux: jnp.ndarray, noise: float) -> jnp.ndarray:
    """
    Add Gaussian noise with std = noise * peak-to-peak(flux).
    """
    amplitude = jnp.maximum(jnp.max(flux) - jnp.min(flux), 1e-6)
    sigma = noise * amplitude
    noise_values = sigma * jax.random.normal(rng_key, shape=flux.shape, dtype=jnp.float32)
    return (flux + noise_values).astype(jnp.float32)


# --- Synthesis Function ---

def synthesize_light_curve(rng_key, sampled_parameters: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """
    Synthesize one light curve from sampled parameters.
    """
    
    noise_key = jax.random.fold_in(rng_key, 1)

    datapoints = sampled_parameters["datapoints"]
    noise = sampled_parameters["noise"]
    inclination = sampled_parameters["inclination"]
    n_spots = sampled_parameters["n_spots"]
    lats = sampled_parameters["lats"]
    lons = sampled_parameters["lons"]
    radii = sampled_parameters["radii"]

    theta_deg = make_theta_grid(datapoints)
    y_spots_dense = build_spot_map(lats, lons, radii)

    flux = flux_from_map_ylm(
        y_dense=y_spots_dense,
        inclination=inclination,
        theta_deg=theta_deg,
    )

    flux_noisy = add_flux_noise(
        rng_key=noise_key,
        flux=flux,
        noise=noise,
    )

    synthesized_data = {
        "datapoints": datapoints,
        "noise": noise,
        "inclination": inclination,
        "n_spots": n_spots,
        "lats": lats,
        "lons": lons,
        "radii": radii,
        "flux": flux,
        "flux_noisy": flux_noisy,
        "theta_deg": theta_deg,
    }

    return synthesized_data