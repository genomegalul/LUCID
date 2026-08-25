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


# --- Utilities ---

def load_curve(npz_path: Path) -> dict:
    with np.load(npz_path, allow_pickle=True) as data:
        curve = {
            "datapoints": int(data["datapoints"]),
            "noise": float(data["noise"]),
            "inclination": float(data["inclination"]),
            "contrast": float(data["contrast"]),
            "n_spots": int(data["n_spots"]),
            "lats": data["lats"].astype(np.float32),
            "lons": data["lons"].astype(np.float32),
            "radii": data["radii"].astype(np.float32),
            "theta_deg": data["theta_deg"].astype(np.float32),
            "flux": data["flux"].astype(np.float32),
            "flux_noisy": data["flux_noisy"].astype(np.float32),
        }
    return curve


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


def make_title(curve: dict, npz_path: Path) -> str:
    return (
        f"{curve['n_spots']}-Spot Light Curve  |  "
        f"incl={curve['inclination']:.1f}°  |  "
        f"contrast={curve['contrast']:.3f}  |  "
        f"noise={curve['noise']:.3f}  |  "
        f"N={curve['datapoints']}\n"
        f"{npz_path.name}"
    )


def plot_curve(curve: dict, npz_path: Path, save_path: Path | None, show: bool) -> None:
    theta = curve["theta_deg"]
    flux = curve["flux"]
    flux_noisy = curve["flux_noisy"]

    plt.figure(figsize=(12, 6))
    plt.plot(theta, flux, linewidth=2.5, label="Clean Flux")
    plt.plot(theta, flux_noisy, linewidth=1.5, alpha=0.8, label="Noisy Flux")

    plt.xlabel("Rotation Phase (degrees)")
    plt.ylabel("Flux")
    plt.title(make_title(curve, npz_path))
    plt.xlim(float(theta.min()), float(theta.max()) if len(theta) > 1 else 360.0)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")

    if show:
        plt.show()

    plt.close()


# --- CLI ---

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot one synthesized light curve, overplotting clean and noisy flux."
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
    curve = load_curve(npz_path)

    save_path = None
    if args.save:
        save_name = npz_path.with_suffix(".png").name
        save_path = OUTPUT_DIR / save_name

    plot_curve(
        curve=curve,
        npz_path=npz_path,
        save_path=save_path,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()