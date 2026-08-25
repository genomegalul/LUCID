from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

# Make project root importable so retrieval2 can use sibling folders cleanly later.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FitConfig, ParameterBounds, config_to_jsonable
from data import load_single_dataset_example, load_curve_file
from likelihood import make_objective
from parameters import physical_params_to_numpy_dict
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
) -> None:
    residual = target_flux - best_flux

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(theta_deg, target_flux, label="Target", lw=2)
    axes[0].plot(theta_deg, best_flux, label="Best fit", lw=2)
    axes[0].set_ylabel("Flux")
    axes[0].legend()

    axes[1].plot(theta_deg, residual, lw=1)
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Theta (deg)")
    axes[1].set_ylabel("Residual")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_loss_plot(
    loss_history: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Best loss")
    plt.title("Retrieval loss history")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a pure-JAX Aeolus-like spot retrieval on one light curve."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "--dataset-file",
        type=str,
        default=None,
        help="Path to single dataset file, e.g. ../data/dataset_256.npz",
    )

    input_group.add_argument(
        "--curve-file",
        type=str,
        default=None,
        help="Standalone .npz file containing flux and theta_deg.",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Example index if using --dataset-file.",
    )

    parser.add_argument(
        "--flux-key",
        type=str,
        default="flux",
        choices=["flux", "flux_noisy"],
    )

    parser.add_argument(
        "--n-spots",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Assumed number of spots for this retrieval.",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="elite_mutation",
        choices=["random_search", "elite_mutation"],
    )

    parser.add_argument("--fit-inclination", action="store_true")
    parser.add_argument("--inclination", type=float, default=None)
    parser.add_argument("--use-true-inclination", action="store_true")

    parser.add_argument("--n-candidates", type=int, default=4096)
    parser.add_argument("--n-elites", type=int, default=256)
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
    datapoints = int(len(target_flux))

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

    config = FitConfig(
        n_spots=args.n_spots,
        datapoints=datapoints,
        fit_inclination=args.fit_inclination,
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
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )

    timestamp = make_timestamp()
    output_dir = Path(args.output_dir) / f"fit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Config:", json.dumps(config_to_jsonable(config), indent=2), flush=True)

    simulate_batch = make_simulator(
        config=config,
        theta_deg=theta_deg,
    )

    score_batch, predict_and_score_batch = make_objective(
        config=config,
        simulate_batch=simulate_batch,
        target_flux=target_flux,
    )

    print("[warmup] compiling simulator/objective...", flush=True)
    test_raw = jnp.zeros((1, 4 * args.n_spots + (1 if args.fit_inclination else 0)), dtype=jnp.float32)
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

    physical_best = physical_params_to_numpy_dict(
        best_raw,
        config=config,
    )
    physical_best = {k: np.asarray(v) for k, v in physical_best.items()}

    final_mse = float(np.mean((best_flux - target_flux) ** 2))

    print("\nFinished retrieval.", flush=True)
    print(f"Best loss: {result.best_loss:.8e}", flush=True)
    print(f"Final MSE: {final_mse:.8e}", flush=True)
    print("Best physical parameters:", flush=True)
    for k, v in physical_best.items():
        print(f"  {k}: {v}", flush=True)

    np.savez(
        output_dir / "result.npz",
        target_flux=target_flux,
        theta_deg=theta_deg,
        best_flux=best_flux,
        best_raw_params=best_raw,
        elite_raw_params=elite_raw,
        elite_losses=elite_losses,
        loss_history=loss_history,
        **physical_best,
    )

    with (output_dir / "summary.json").open("w") as f:
        json.dump(
            {
                "strategy": args.strategy,
                "best_loss": float(result.best_loss),
                "final_mse": final_mse,
                "config": config_to_jsonable(config),
                "input": {
                    "dataset_file": args.dataset_file,
                    "curve_file": args.curve_file,
                    "index": args.index,
                    "flux_key": args.flux_key,
                    "true_n_spots": example.get("n_spots"),
                    "true_inclination": example.get("inclination"),
                },
            },
            f,
            indent=2,
        )

    save_fit_plot(
        theta_deg=theta_deg,
        target_flux=target_flux,
        best_flux=best_flux,
        output_path=output_dir / "fit.png",
    )

    save_loss_plot(
        loss_history=loss_history,
        output_path=output_dir / "loss.png",
    )

    print(f"\nSaved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()