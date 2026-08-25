from __future__ import annotations

import argparse
import json
import sys
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

from config import FitConfig, ParameterBounds
from simulator import make_simulator


def config_from_summary(summary: dict) -> FitConfig:
    cfg = summary["config"]
    bounds = ParameterBounds(**cfg["bounds"])

    return FitConfig(
        n_spots=int(cfg["n_spots"]),
        datapoints=int(cfg["datapoints"]),
        fit_inclination=bool(cfg["fit_inclination"]),
        fixed_inclination_deg=cfg["fixed_inclination_deg"],
        bounds=bounds,
        raw_init_scale=float(cfg["raw_init_scale"]),
        raw_mutation_scale=float(cfg["raw_mutation_scale"]),
        use_separation_penalty=bool(cfg["use_separation_penalty"]),
        min_spot_gap_deg=float(cfg["min_spot_gap_deg"]),
        separation_penalty_weight=float(cfg["separation_penalty_weight"]),
        n_candidates=int(cfg["n_candidates"]),
        n_elites=int(cfg["n_elites"]),
        n_steps=int(cfg["n_steps"]),
        seed=int(cfg["seed"]),
        output_dir=Path(cfg["output_dir"]),
    )


def simulate_elite_curves(
    elite_raw_params: np.ndarray,
    theta_deg: np.ndarray,
    config: FitConfig,
    n_elites_to_plot: int,
) -> np.ndarray:
    n = min(n_elites_to_plot, elite_raw_params.shape[0])
    raw = jnp.asarray(elite_raw_params[:n], dtype=jnp.float32)

    simulate_batch = make_simulator(
        config=config,
        theta_deg=theta_deg,
    )

    print(f"[plot] simulating {n} elite curves for uncertainty...", flush=True)
    curves = simulate_batch(raw)
    curves.block_until_ready()

    return np.asarray(curves, dtype=np.float32)


def plot_result(
    output_dir: Path,
    n_elites_to_plot: int,
    n_sample_lines: int,
    percentile_low: float,
    percentile_high: float,
) -> None:
    result_path = output_dir / "result.npz"
    summary_path = output_dir / "summary.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Could not find result file: {result_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Could not find summary file: {summary_path}")

    with summary_path.open("r") as f:
        summary = json.load(f)

    config = config_from_summary(summary)

    with np.load(result_path, allow_pickle=False) as data:
        theta_deg = data["theta_deg"].astype(np.float32)
        target_flux = data["target_flux"].astype(np.float32)
        best_flux = data["best_flux"].astype(np.float32)
        elite_raw_params = data["elite_raw_params"].astype(np.float32)
        elite_losses = data["elite_losses"].astype(np.float32)
        loss_history = data["loss_history"].astype(np.float32)

    elite_curves = simulate_elite_curves(
        elite_raw_params=elite_raw_params,
        theta_deg=theta_deg,
        config=config,
        n_elites_to_plot=n_elites_to_plot,
    )

    mean_curve = elite_curves.mean(axis=0)
    std_curve = elite_curves.std(axis=0)

    lo_curve = np.percentile(elite_curves, percentile_low, axis=0)
    hi_curve = np.percentile(elite_curves, percentile_high, axis=0)

    residual_best = target_flux - best_flux
    residual_mean = target_flux - mean_curve

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2.5, 1.2, 1.2]},
    )

    ax = axes[0]

    if n_sample_lines > 0:
        n_lines = min(n_sample_lines, elite_curves.shape[0])
        for i in range(n_lines):
            ax.plot(
                theta_deg,
                elite_curves[i],
                linewidth=0.7,
                alpha=0.15,
            )

    ax.fill_between(
        theta_deg,
        lo_curve,
        hi_curve,
        alpha=0.25,
        label=f"{percentile_low:.0f}-{percentile_high:.0f}% elite band",
    )

    ax.plot(theta_deg, target_flux, linewidth=2.0, label="Target")
    ax.plot(theta_deg, best_flux, linewidth=2.0, linestyle="--", label="Best fit")
    ax.plot(theta_deg, mean_curve, linewidth=1.5, linestyle=":", label="Elite mean")

    ax.set_ylabel("Flux")
    ax.set_title(
        f"{config.n_spots}-spot retrieval | "
        f"best MSE={summary['final_mse']:.3e}"
    )
    ax.legend()

    axes[1].plot(theta_deg, residual_best, linewidth=1.0)
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Target - best")

    axes[2].plot(theta_deg, residual_mean, linewidth=1.0)
    axes[2].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[2].set_ylabel("Target - mean")
    axes[2].set_xlabel("Theta (deg)")

    fig.tight_layout()

    plot_path = output_dir / "fit_uncertainty.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(loss_history)), loss_history)
    plt.yscale("log")
    plt.xlabel("Step")
    plt.ylabel("Best loss")
    plt.title("Retrieval loss history")
    plt.tight_layout()

    loss_path = output_dir / "loss_history.png"
    plt.savefig(loss_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Elite loss histogram
    plt.figure(figsize=(8, 5))
    plt.hist(elite_losses, bins=40)
    plt.yscale("log")
    plt.xlabel("Elite loss")
    plt.ylabel("Count")
    plt.title("Elite solution loss distribution")
    plt.tight_layout()

    hist_path = output_dir / "elite_loss_hist.png"
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Save uncertainty arrays
    np.savez(
        output_dir / "uncertainty_curves.npz",
        theta_deg=theta_deg,
        target_flux=target_flux,
        best_flux=best_flux,
        elite_curves=elite_curves,
        mean_curve=mean_curve,
        std_curve=std_curve,
        lo_curve=lo_curve,
        hi_curve=hi_curve,
        percentile_low=np.asarray(percentile_low, dtype=np.float32),
        percentile_high=np.asarray(percentile_high, dtype=np.float32),
    )

    print(f"Saved uncertainty plot: {plot_path}", flush=True)
    print(f"Saved loss plot:        {loss_path}", flush=True)
    print(f"Saved loss histogram:   {hist_path}", flush=True)
    print(f"Saved uncertainty data: {output_dir / 'uncertainty_curves.npz'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an aeolus_like retrieval result with elite-solution uncertainty."
    )

    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to a retrieval output directory containing result.npz and summary.json.",
    )

    parser.add_argument(
        "--n-elites",
        type=int,
        default=128,
        help="Number of elite solutions to re-simulate for uncertainty.",
    )

    parser.add_argument(
        "--sample-lines",
        type=int,
        default=25,
        help="Number of individual elite sample curves to overplot faintly.",
    )

    parser.add_argument(
        "--percentile-low",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--percentile-high",
        type=float,
        default=95.0,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("JAX devices:", jax.devices(), flush=True)
    print("JAX backend:", jax.default_backend(), flush=True)

    plot_result(
        output_dir=Path(args.output_dir).expanduser().resolve(),
        n_elites_to_plot=args.n_elites,
        n_sample_lines=args.sample_lines,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
    )


if __name__ == "__main__":
    main()