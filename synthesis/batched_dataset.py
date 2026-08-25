from __future__ import annotations

import os
from typing import Dict

import jax
import numpy as np
from tqdm import tqdm

from sampling import DATAPOINTS, sample_parameters_batch
from simulator import synthesize_light_curve_batch


# --- Dataset Configuration ---

OUTPUT_DIR = "../data"
SPOT_COUNTS = [1, 2, 3, 4, 5]
MAX_SPOTS = max(SPOT_COUNTS)

BATCH_SIZE = 4096
N_DATASETS = 250 * BATCH_SIZE
SEED = 0

OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"dataset_{DATAPOINTS}.npz")


# --- Utilities ---

def _to_numpy(x):
    return np.asarray(x)


def _allocate_arrays() -> Dict[str, np.ndarray]:
    arrays = {
        "flux": np.empty((N_DATASETS, DATAPOINTS), dtype=np.float32),
        "flux_noisy": np.empty((N_DATASETS, DATAPOINTS), dtype=np.float32),

        "n_spots": np.empty(N_DATASETS, dtype=np.int32),
        "labels": np.empty(N_DATASETS, dtype=np.int64),

        "noise": np.empty(N_DATASETS, dtype=np.float32),
        "inclination": np.empty(N_DATASETS, dtype=np.float32),
        "contrast": np.empty(N_DATASETS, dtype=np.float32),

        "lats": np.full((N_DATASETS, MAX_SPOTS), np.nan, dtype=np.float32),
        "lons": np.full((N_DATASETS, MAX_SPOTS), np.nan, dtype=np.float32),
        "radii": np.full((N_DATASETS, MAX_SPOTS), np.nan, dtype=np.float32),
    }
    return arrays


def _save_single_dataset_file(
    filename: str,
    arrays: Dict[str, np.ndarray],
    theta_deg: np.ndarray,
    n_per_category: int,
) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    tmp_filename = filename + ".tmp"

    np.savez(
        tmp_filename,
        datapoints=np.asarray(DATAPOINTS, dtype=np.int32),
        n_datasets=np.asarray(N_DATASETS, dtype=np.int32),
        n_per_category=np.asarray(n_per_category, dtype=np.int32),
        spot_counts=np.asarray(SPOT_COUNTS, dtype=np.int32),
        max_spots=np.asarray(MAX_SPOTS, dtype=np.int32),
        seed=np.asarray(SEED, dtype=np.int32),

        theta_deg=theta_deg.astype(np.float32),

        flux=arrays["flux"],
        flux_noisy=arrays["flux_noisy"],

        n_spots=arrays["n_spots"],
        labels=arrays["labels"],

        noise=arrays["noise"],
        inclination=arrays["inclination"],
        contrast=arrays["contrast"],

        lats=arrays["lats"],
        lons=arrays["lons"],
        radii=arrays["radii"],
    )

    os.replace(tmp_filename + ".npz", filename)


# --- Driver ---

def main():
    print("JAX devices:", jax.devices(), flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if os.path.exists(OUTPUT_FILE):
        raise FileExistsError(
            f"Output file already exists: {OUTPUT_FILE}\n"
            f"Delete it first if you want to regenerate the dataset."
        )

    n_categories = len(SPOT_COUNTS)
    if N_DATASETS % n_categories != 0:
        raise ValueError(
            f"N_DATASETS ({N_DATASETS}) must be evenly divisible by "
            f"the number of spot categories ({n_categories})."
        )

    n_per_category = N_DATASETS // n_categories

    arrays = _allocate_arrays()
    theta_deg = None

    master_key = jax.random.PRNGKey(SEED)

    print(f"[seed] master seed = {SEED}", flush=True)
    print(f"[batch] batch size = {BATCH_SIZE}", flush=True)
    print(f"[output] {OUTPUT_FILE}", flush=True)
    print(f"[datapoints] {DATAPOINTS}", flush=True)
    print(f"[total] {N_DATASETS}", flush=True)
    print(f"[per category] {n_per_category}", flush=True)

    for n_spots in SPOT_COUNTS:
        category_index = SPOT_COUNTS.index(n_spots)
        start_index = category_index * n_per_category
        end_index = start_index + n_per_category

        print(
            f"[plan] spots={n_spots} -> range [{start_index}, {end_index})",
            flush=True,
        )

        for batch_start in tqdm(
            range(start_index, end_index, BATCH_SIZE),
            desc=f"{n_spots}spot",
            unit="batch",
        ):
            batch_end = min(batch_start + BATCH_SIZE, end_index)
            batch_indices = np.arange(batch_start, batch_end, dtype=np.int64)

            example_ids = jax.numpy.asarray(batch_indices, dtype=jax.numpy.uint32)

            example_keys = jax.vmap(
                lambda idx: jax.random.fold_in(master_key, n_spots * 1_000_000 + idx)
            )(example_ids)

            sample_keys = jax.vmap(lambda k: jax.random.fold_in(k, 0))(example_keys)
            rng_keys = jax.vmap(lambda k: jax.random.fold_in(k, 1))(example_keys)

            sampled_parameters_batch = sample_parameters_batch(
                sample_keys,
                n_spots=n_spots,
            )

            synthesized_batch = synthesize_light_curve_batch(
                rng_keys,
                sampled_parameters_batch,
            )

            synthesized_batch = jax.device_get(synthesized_batch)

            idx = batch_indices
            batch_size = len(idx)

            arrays["flux"][idx] = _to_numpy(synthesized_batch["flux"]).astype(np.float32)
            arrays["flux_noisy"][idx] = _to_numpy(synthesized_batch["flux_noisy"]).astype(np.float32)

            if theta_deg is None:
                theta_deg = _to_numpy(synthesized_batch["theta_deg"][0]).astype(np.float32)

            n_spots_batch = _to_numpy(synthesized_batch["n_spots"]).astype(np.int32)
            arrays["n_spots"][idx] = n_spots_batch
            arrays["labels"][idx] = n_spots_batch.astype(np.int64) - 1

            arrays["noise"][idx] = _to_numpy(synthesized_batch["noise"]).astype(np.float32)
            arrays["inclination"][idx] = _to_numpy(synthesized_batch["inclination"]).astype(np.float32)
            arrays["contrast"][idx] = _to_numpy(synthesized_batch["contrast"]).astype(np.float32)

            lats = _to_numpy(synthesized_batch["lats"]).astype(np.float32)
            lons = _to_numpy(synthesized_batch["lons"]).astype(np.float32)
            radii = _to_numpy(synthesized_batch["radii"]).astype(np.float32)

            arrays["lats"][idx, :n_spots] = lats.reshape(batch_size, n_spots)
            arrays["lons"][idx, :n_spots] = lons.reshape(batch_size, n_spots)
            arrays["radii"][idx, :n_spots] = radii.reshape(batch_size, n_spots)

    if theta_deg is None:
        raise RuntimeError("No data were generated, so theta_deg was never initialized.")

    print(f"[save] writing single dataset file: {OUTPUT_FILE}", flush=True)
    _save_single_dataset_file(
        filename=OUTPUT_FILE,
        arrays=arrays,
        theta_deg=theta_deg,
        n_per_category=n_per_category,
    )

    print("Done.", flush=True)


if __name__ == "__main__":
    main()