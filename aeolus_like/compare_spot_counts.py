from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FitConfig, ParameterBounds, config_to_jsonable
from data import load_single_dataset_example, load_curve_file
from likelihood import make_objective
from parameters import n_raw_params, physical_params_to_numpy_dict
from simulator import make_simulator
from strategies.random_search import run_random_search
from strategies.elite_mutation import run_elite_mutation


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_fit_plot(
    theta_deg: np.ndarray,
    target_flux: np.ndarray,
    best_flux: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    residual = target_flux - best_flux

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(theta_deg, target_flux, label="Target", lw=2)
    axes[0].plot(theta_deg, best_flux, label="Best fit", lw=2, linestyle="--")
    axes[0].set_ylabel("Flux")
    axes[0].set_title(title)
    axes[0].legend()

    axes[1].plot(theta_deg, residual, lw=1)
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Theta (deg)")
    axes[1].set_ylabel("Residual")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_comparison_plot(
    rows: list[dict],
    output_path: Path,
) -> None:
    n_spots = np.array([row["n_spots"] for row in rows], dtype=np.int32)
    mse = np.array([row["mse"] for row in rows], dtype=np.float64)
    aic = np.array([row["aic"] for row in rows], dtype=np.float64)
    bic = np.array([row["bic"] for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    axes[0].plot(n_spots, mse, marker="o")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Best MSE")
    axes[0].set_title("Spot-count comparison")

    axes[1].plot(n_spots, aic, marker="o")
    axes[1].set_ylabel("AIC-like score")

    axes[2].plot(n_spots, bic, marker="o")
    axes[2].set_ylabel("BIC-like score")
    axes[2].set_xlabel("Assumed number of spots")
    axes[2].set_xticks(n_spots)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_information_scores(
    mse: float,
    n_params: int,
    n_points: int,
) -> tuple[float, float]:
    """
    Gaussian residual model with unknown constant variance, up to additive constants:

        AIC = N log(MSE) + 2k
        BIC = N log(MSE) + k log(N)

    Lower is better.
    """
    safe_mse = max(float(mse), 1e-30)
    aic = n_points * np.log(safe_mse) + 2.0 * n_params
    bic = n_points * np.log(safe_mse) + n_params * np.log(n_points)
    return float(aic), float(bic)


def run_one_spot_count(
    n_spots: int,
    target_flux: np.ndarray,
    theta_deg: np.ndarray,
    fixed_inclination_deg: float | None,
    fit_inclination: bool,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    config = FitConfig(
        n_spots=n_spots,
        datapoints=len(target_flux),
        fit_inclination=fit_inclination,
        fixed_inclination_deg=fixed_inclination_deg,
        bounds=ParameterBounds(),
        raw_init_scale=args.raw_init_scale,
        raw_mutation_scale=args.raw_mutation_scale,
        use_separation_penalty=not args.no_separation_penalty,
        min_spot_gap_deg=args.min_spot_gap_deg,
        separation_penalty_weight=args.separation_penalty_weight,
        n_candidates=args.n_candidates,
        n_elites=args.n_elites,
        n_steps=args.n_steps,
        seed=args.seed + 10_000 * n_spots,
        output_dir=output_dir,
    )

    print("=" * 80, flush=True)
    print(f"Running assumed n_spots={n_spots}", flush=True)
    print(json.dumps(config_to_jsonable(config), indent=2), flush=True)

    simulate_batch = make_simulator(
        config=config,
        theta_deg=theta_deg,
    )

    score_batch, predict_and_score_batch = make_objective(
        config=config,
        simulate_batch=simulate_batch,
        target_flux=target_flux,
    )

    raw_dim = n_raw_params(config)

    print("[warmup] compiling objective...", flush=True)
    test_raw = jnp.zeros((1, raw_dim), dtype=jnp.float32)
    test_pred, test_loss = predict_and_score_batch(test_raw)
    test_pred.block_until_ready()
    test_loss.block_until_ready()
    print("[warmup] done.", flush=True)

    if args.strategy == "random_search":
        result = run_random_search(
            config=config,
            score_batch=score_batch,
            predict_and_score_batch=predict_and_score_batch,
        )
    elif args.strategy == "elite_mutation":
        result = run_elite_mutation(
            config=config,
            score_batch=score_batch,
            predict_and_score_batch=predict_and_score_batch,
        )
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    best_raw = np.asarray(result.best_raw_params)
    best_flux = np.asarray(result.best_flux)
    elite_raw = np.asarray(result.elite_raw_params)
    elite_losses = np.asarray(result.elite_losses)
    loss_history = np.asarray(result.loss_history)

    mse = float(np.mean((best_flux - target_flux) ** 2))
    aic, bic = compute_information_scores(
        mse=mse,
        n_params=raw_dim,
        n_points=len(target_flux),
    )

    physical_best = physical_params_to_numpy_dict(
        best_raw,
        config=config,
    )
    physical_best = {k: np.asarray(v) for k, v in physical_best.items()}

    spot_dir = output_dir / f"{n_spots}spot"
    spot_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        spot_dir / "result.npz",
        target_flux=target_flux,
        theta_deg=theta_deg,
        best_flux=best_flux,
        best_raw_params=best_raw,
        elite_raw_params=elite_raw,
        elite_losses=elite_losses,
        loss_history=loss_history,
        **physical_best,
    )

    with (spot_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "strategy": args.strategy,
                "n_spots": n_spots,
                "best_loss": float(result.best_loss),
                "mse": mse,
                "aic": aic,
                "bic": bic,
                "n_params": raw_dim,
                "config": config_to_jsonable(config),
            },
            f,
            indent=2,
        )

    save_fit_plot(
        theta_deg=theta_deg,
        target_flux=target_flux,
        best_flux=best_flux,
        output_path=spot_dir / "fit.png",
        title=f"{n_spots}-spot fit | MSE={mse:.3e}",
    )

    print(f"n_spots={n_spots} MSE={mse:.8e} AIC={aic:.3f} BIC={bic:.3f}", flush=True)

    return {
        "n_spots": n_spots,
        "mse": mse,
        "best_loss": float(result.best_loss),
        "aic": aic,
        "bic": bic,
        "n_params": raw_dim,
        "output_dir": str(spot_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run aeolus_like retrieval for multiple assumed spot counts and compare fits."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--dataset-file",
        type=str,
        default=None,
    )

    input_group.add_argument(
        "--curve-file",
        type=str,
        default=None,
    )

    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--flux-key", type=str, default="flux", choices=["flux", "flux_noisy"])

    parser.add_argument("--min-spots", type=int, default=1)
    parser.add_argument("--max-spots", type=int, default=5)

    parser.add_argument(
        "--strategy",
        type=str,
        default="elite_mutation",
        choices=["random_search", "elite_mutation"],
    )

    parser.add_argument("--fit-inclination", action="store_true")
    parser.add_argument("--inclination", type=float, default=None)
    parser.add_argument("--use-true-inclination", action="store_true")

    parser.add_argument("--n-candidates", type=int, default=8192)
    parser.add_argument("--n-elites", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=250)

    parser.add_argument("--raw-init-scale", type=float, default=2.0)
    parser.add_argument("--raw-mutation-scale", type=float, default=0.35)

    parser.add_argument("--no-separation-penalty", action="store_true")
    parser.add_argument("--min-spot-gap-deg", type=float, default=2.0)
    parser.add_argument("--separation-penalty-weight", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_spots < 1 or args.max_spots > 5 or args.min_spots > args.max_spots:
        raise ValueError("Require 1 <= min_spots <= max_spots <= 5")

    print("JAX devices:", jax.devices(), flush=True)
    print("JAX backend:", jax.default_backend(), flush=True)

    if args.dataset_file is not None:
        example = load_single_dataset_example(
            dataset_file=args.dataset_file,
            index=args.index,
            flux_key=args.flux_key,
        )
    else:
        example = load_curve_file(args.curve_file)

    target_flux = example["flux"].astype(np.float32)
    theta_deg = example["theta_deg"].astype(np.float32)

    if args.fit_inclination:
        fixed_inclination_deg = None
    elif args.use_true_inclination:
        if example.get("inclination") is None:
            raise ValueError("Input does not contain inclination, so --use-true-inclination cannot be used.")
        fixed_inclination_deg = float(example["inclination"])
    elif args.inclination is not None:
        fixed_inclination_deg = float(args.inclination)
    else:
        raise ValueError(
            "Provide one of: --fit-inclination, --use-true-inclination, or --inclination VALUE"
        )

    timestamp = make_timestamp()

    output_dir = Path(args.output_dir) / f"compare_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for n_spots in range(args.min_spots, args.max_spots + 1):
        row = run_one_spot_count(
            n_spots=n_spots,
            target_flux=target_flux,
            theta_deg=theta_deg,
            fixed_inclination_deg=fixed_inclination_deg,
            fit_inclination=args.fit_inclination,
            args=args,
            output_dir=output_dir,
        )
        rows.append(row)

    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "n_spots",
                "mse",
                "best_loss",
                "aic",
                "bic",
                "n_params",
                "output_dir",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with (output_dir / "comparison.json").open("w") as f:
        json.dump(
            {
                "input": {
                    "dataset_file": args.dataset_file,
                    "curve_file": args.curve_file,
                    "index": args.index,
                    "flux_key": args.flux_key,
                    "true_n_spots": example.get("n_spots"),
                    "true_inclination": example.get("inclination"),
                },
                "strategy": args.strategy,
                "rows": rows,
            },
            f,
            indent=2,
        )

    save_comparison_plot(
        rows=rows,
        output_path=output_dir / "comparison.png",
    )

    best_by_mse = min(rows, key=lambda r: r["mse"])
    best_by_aic = min(rows, key=lambda r: r["aic"])
    best_by_bic = min(rows, key=lambda r: r["bic"])

    print("\nComparison complete.", flush=True)
    print(f"Saved to: {output_dir}", flush=True)
    print(f"Best by MSE: n_spots={best_by_mse['n_spots']} mse={best_by_mse['mse']:.8e}", flush=True)
    print(f"Best by AIC: n_spots={best_by_aic['n_spots']} aic={best_by_aic['aic']:.3f}", flush=True)
    print(f"Best by BIC: n_spots={best_by_bic['n_spots']} bic={best_by_bic['bic']:.3f}", flush=True)


if __name__ == "__main__":
    main()