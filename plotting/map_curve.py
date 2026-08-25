from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# --- Defaults ---

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = SCRIPT_DIR / "plots"

MERCATOR_LAT_LIMIT_DEG = 85.0


# --- Utilities ---

def load_map_parameters(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=True) as data:
        result = {
            "datapoints": int(data["datapoints"]),
            "noise": float(data["noise"]),
            "inclination": float(data["inclination"]),
            "contrast": float(data["contrast"]),
            "n_spots": int(data["n_spots"]),
            "lats": data["lats"].astype(np.float32),
            "lons": data["lons"].astype(np.float32),
            "radii": data["radii"].astype(np.float32),
        }
    return result


def resolve_npz_path(file_path: str | None, spots: int | None, index: int | None) -> Path:
    if file_path is not None:
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Could not find file: {path}")
        return path

    if spots is None or index is None:
        raise ValueError("Provide either --file or both --spots and --index.")

    path = (DATA_DIR / f"{spots}spot" / f"dataset_{index:07d}.npz").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Could not find file: {path}")
    return path


def sph_to_cart(lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return x, y, z


def mercator_y(lat_deg: np.ndarray) -> np.ndarray:
    lat_deg = np.clip(lat_deg, -MERCATOR_LAT_LIMIT_DEG, MERCATOR_LAT_LIMIT_DEG)
    lat_rad = np.deg2rad(lat_deg)
    return np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))


def build_surface_map(curve: dict, n_lat: int = 361, n_lon: int = 721) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a latitude-longitude intensity map from the saved spot parameters.

    The baseline intensity is 1.0. Each circular spot subtracts the saved
    per-map contrast wherever the great-circle angular separation from the
    spot center is less than the spot radius.
    """
    lat_grid = np.linspace(-MERCATOR_LAT_LIMIT_DEG, MERCATOR_LAT_LIMIT_DEG, n_lat, dtype=np.float32)
    lon_grid = np.linspace(-180.0, 180.0, n_lon, dtype=np.float32)

    lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
    gx, gy, gz = sph_to_cart(lat2d, lon2d)

    intensity = np.ones_like(lat2d, dtype=np.float32)
    contrast = float(curve["contrast"])

    for lat0, lon0, radius0 in zip(curve["lats"], curve["lons"], curve["radii"]):
        sx, sy, sz = sph_to_cart(np.asarray(lat0), np.asarray(lon0))
        cos_sep = gx * sx + gy * sy + gz * sz
        cos_sep = np.clip(cos_sep, -1.0, 1.0)
        sep_deg = np.rad2deg(np.arccos(cos_sep))
        mask = sep_deg <= float(radius0)
        intensity[mask] -= contrast

    intensity = np.clip(intensity, 0.0, 1.0)
    return lat_grid, lon_grid, intensity


def make_title(curve: dict, npz_path: Path) -> str:
    return (
        f"{curve['n_spots']}-Spot Atmospheric Map  |  "
        f"incl={curve['inclination']:.1f}°  |  "
        f"contrast={curve['contrast']:.3f}  |  "
        f"noise={curve['noise']:.3f}  |  "
        f"N={curve['datapoints']}\n"
        f"{npz_path.name}"
    )


def plot_map(curve: dict, npz_path: Path, save_path: Path | None, show: bool) -> None:
    lat_grid, lon_grid, intensity = build_surface_map(curve)
    y_grid = mercator_y(lat_grid)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        intensity,
        extent=[lon_grid.min(), lon_grid.max(), y_grid.min(), y_grid.max()],
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )

    tick_lats = np.array([-80, -60, -30, 0, 30, 60, 80], dtype=np.float32)
    tick_ys = mercator_y(tick_lats)

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (Mercator)")
    ax.set_yticks(tick_ys)
    ax.set_yticklabels([f"{lat:.0f}°" for lat in tick_lats])
    ax.set_title(make_title(curve, npz_path))

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Surface Intensity")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# --- CLI ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one synthesized atmospheric map in Mercator projection from a dataset .npz file."
    )

    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Direct path to a dataset .npz file.",
    )
    parser.add_argument(
        "--spots",
        type=int,
        default=None,
        help="Spot-count directory to use (1-5) when resolving from the default data folder.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Dataset index when resolving from the default data folder.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the figure to ../plotting/plots/ instead of only showing it.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    npz_path = resolve_npz_path(args.file, args.spots, args.index)
    curve = load_map_parameters(npz_path)

    save_path = None
    if args.save:
        save_name = npz_path.with_suffix("").name + "_map_mercator.png"
        save_path = OUTPUT_DIR / save_name

    plot_map(
        curve=curve,
        npz_path=npz_path,
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()