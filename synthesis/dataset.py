from __future__ import annotations

import os
from typing import Dict

import jax
import numpy as np
from tqdm import tqdm

from sampling import sample_parameters
from simulator import synthesize_light_curve


# --- Dataset Configuration ---

OUTPUT_DIR = "../data"
SPOT_COUNTS = [1, 2, 3, 4, 5]
N_DATASETS = 100_000
SEED = 0


# --- IO Utilities ---

def _to_numpy(x):
    return np.asarray(x)


def _scalarize(x):
    arr = np.asarray(x)
    if arr.ndim == 0:
        if arr.dtype.kind in ("i", "u"):
            return int(arr)
        if arr.dtype.kind == "f":
            return float(arr)
    return arr


def _save_dataset(filename: str, synthesized_data: Dict):
    """
    Save one synthesized example to a compressed .npz file.
    """
    tmp_filename = filename + ".tmp.npz"

    payload = {
        "datapoints": _scalarize(synthesized_data["datapoints"]),
        "noise": _scalarize(synthesized_data["noise"]),
        "inclination": _scalarize(synthesized_data["inclination"]),
        "contrast": _scalarize(synthesized_data["contrast"]),
        "n_spots": _scalarize(synthesized_data["n_spots"]),
        "lats": _to_numpy(synthesized_data["lats"]).astype(np.float32),
        "lons": _to_numpy(synthesized_data["lons"]).astype(np.float32),
        "radii": _to_numpy(synthesized_data["radii"]).astype(np.float32),
        "flux": _to_numpy(synthesized_data["flux"]).astype(np.float32),
        "flux_noisy": _to_numpy(synthesized_data["flux_noisy"]).astype(np.float32),
        "theta_deg": _to_numpy(synthesized_data["theta_deg"]).astype(np.float32),
    }

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(tmp_filename, **payload)
    os.replace(tmp_filename, filename)


# --- Driver ---

def main():
    print("JAX devices:", jax.devices(), flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_categories = len(SPOT_COUNTS)
    if N_DATASETS % n_categories != 0:
        raise ValueError(
            f"N_DATASETS ({N_DATASETS}) must be evenly divisible by "
            f"the number of spot categories ({n_categories})."
        )

    n_per_category = N_DATASETS // n_categories

    for n_spots in SPOT_COUNTS:
        os.makedirs(os.path.join(OUTPUT_DIR, f"{n_spots}spot"), exist_ok=True)

    master_key = jax.random.PRNGKey(SEED)
    print(f"[seed] master seed = {SEED}", flush=True)

    for n_spots in SPOT_COUNTS:
        category_index = SPOT_COUNTS.index(n_spots)
        start_index = category_index * n_per_category
        end_index = start_index + n_per_category
        outdir = os.path.join(OUTPUT_DIR, f"{n_spots}spot")

        print(
            f"[plan] spots={n_spots} -> range [{start_index}, {end_index})",
            flush=True,
        )

        for i in tqdm(
            range(start_index, end_index),
            desc=f"{n_spots}spot",
            unit="samples",
        ):
            filename = os.path.join(outdir, f"dataset_{i:07d}.npz")
            if os.path.exists(filename):
                continue

            rng_key = jax.random.fold_in(master_key, (n_spots * 1_000_000) + i)

            sampled_parameters = sample_parameters(rng_key, n_spots)
            synthesized_data = synthesize_light_curve(rng_key, sampled_parameters)

            jax.tree.map(
                lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
                synthesized_data,
            )

            _save_dataset(filename, synthesized_data)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()