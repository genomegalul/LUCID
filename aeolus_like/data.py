from __future__ import annotations

from pathlib import Path

import numpy as np


def load_single_dataset_example(
    dataset_file: str | Path,
    index: int,
    flux_key: str = "flux",
) -> dict:
    """
    Load one example from the new single-file dataset.

    Expected file layout:
        data/dataset_256.npz
        data/dataset_128.npz
        etc.

    Required arrays:
        flux or flux_noisy
        theta_deg
        n_spots
        inclination
        contrast
        noise
        lats
        lons
        radii
    """
    dataset_file = Path(dataset_file).expanduser().resolve()

    if not dataset_file.exists():
        raise FileNotFoundError(f"Could not find dataset file: {dataset_file}")

    if flux_key not in ("flux", "flux_noisy"):
        raise ValueError(f"flux_key must be 'flux' or 'flux_noisy', got {flux_key}")

    with np.load(dataset_file, allow_pickle=False) as data:
        flux = data[flux_key][index].astype(np.float32)
        theta_deg = data["theta_deg"].astype(np.float32)

        result = {
            "dataset_file": str(dataset_file),
            "index": int(index),
            "flux_key": flux_key,
            "flux": flux,
            "theta_deg": theta_deg,
            "datapoints": int(data["datapoints"]),
            "n_spots": int(data["n_spots"][index]),
            "label": int(data["labels"][index]),
            "inclination": float(data["inclination"][index]),
            "contrast": float(data["contrast"][index]),
            "noise": float(data["noise"][index]),
            "lats": data["lats"][index].astype(np.float32),
            "lons": data["lons"][index].astype(np.float32),
            "radii": data["radii"][index].astype(np.float32),
        }

    return result


def load_curve_file(
    curve_file: str | Path,
) -> dict:
    """
    Load a standalone curve file.

    Supported simple .npz layout:
        flux
        theta_deg
        optional inclination
    """
    curve_file = Path(curve_file).expanduser().resolve()

    if not curve_file.exists():
        raise FileNotFoundError(f"Could not find curve file: {curve_file}")

    with np.load(curve_file, allow_pickle=False) as data:
        flux = data["flux"].astype(np.float32)
        theta_deg = data["theta_deg"].astype(np.float32)

        result = {
            "curve_file": str(curve_file),
            "flux": flux,
            "theta_deg": theta_deg,
            "datapoints": int(len(flux)),
            "inclination": float(data["inclination"]) if "inclination" in data else None,
        }

    return result


def normalize_curve_mean(flux: np.ndarray) -> np.ndarray:
    """
    Mean-center the curve for fitting if desired.

    For pure forward-model fitting, you may not want to normalize.
    This function is here for optional later use.
    """
    flux = flux.astype(np.float32)
    return flux - flux.mean(dtype=np.float32)