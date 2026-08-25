from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
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
from data import load_single_dataset_example
from parameters import (
    n_raw_params,
    physical_params_to_numpy_dict,
    spot_separation_penalty,
)
from simulator import make_simulator
from strategies.random_search import run_random_search
from strategies.elite_mutation import run_elite_mutation


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_information_scores(
    mse: float,
    n_params: int,
    n_points: int,
) -> tuple[float, float]:
    safe_mse = max(float(mse), 1e-30)
    aic = n_points * np.log(safe_mse) + 2.0 * n_params
    bic = n_points * np.log(safe_mse) + n_params * np.log(n_points)
    return float(aic), float(bic)


def choose_example_indices(
    dataset_file: Path,
    examples_per_class: int,
    seed: int,
    selection: str,
) -> list[dict]:
    with np.load(dataset_file, allow_pickle=False) as data:
        spot_counts = [int(x) for x in data["spot_counts"]]
        n_per_category = int(data["n_per_category"])

    rng = np.random.default_rng(seed)
    records = []

    for class_i, true_n_spots in enumerate(spot_counts):
        class_start = class_i * n_per_category

        if examples_per_class > n_per_category:
            raise ValueError(
                f"examples_per_class={examples_per_class} exceeds class size {n_per_category}"
            )

        if selection == "evenly_spaced":
            offsets = np.linspace(
                0,
                n_per_category - 1,
                examples_per_class,
                dtype=np.int64,
            )
        elif selection == "random":
            offsets = rng.choice(
                n_per_category,
                size=examples_per_class,
                replace=False,
            )
            offsets = np.sort(offsets)
        else:
            raise ValueError(f"Unknown selection mode: {selection}")

        for offset in offsets:
            records.append(
                {
                    "true_n_spots": int(true_n_spots),
                    "index": int(class_start + offset),
                    "class_offset": int(offset),
                }
            )

    return records


def make_reusable_objective(
    config: FitConfig,
    simulate_batch,
):
    """
    Build objective functions that do not close over target_flux or inclination.

    This is the key recompilation fix.

    These functions can be reused across all examples for a fixed:
        assumed n_spots,
        datapoints,
        fit_inclination mode,
        candidate batch shape.

    Runtime inputs:
        raw_params_batch
        target_flux
        inclination_deg
    """

    @jax.jit
    def score_batch(
        raw_params_batch: jnp.ndarray,
        target_flux: jnp.ndarray,
        inclination_deg: jnp.ndarray,
    ) -> jnp.ndarray:
        pred_batch = simulate_batch(
            raw_params_batch,
            inclination_deg,
        )

        losses = jnp.mean(
            (pred_batch - target_flux[None, :]) ** 2,
            axis=1,
        )

        if config.use_separation_penalty:
            penalties = jax.vmap(
                lambda p: spot_separation_penalty(p, config)
            )(raw_params_batch)

            losses = losses + config.separation_penalty_weight * penalties

        return losses.astype(jnp.float32)

    @jax.jit
    def predict_and_score_batch(
        raw_params_batch: jnp.ndarray,
        target_flux: jnp.ndarray,
        inclination_deg: jnp.ndarray,
    ):
        pred_batch = simulate_batch(
            raw_params_batch,
            inclination_deg,
        )

        losses = jnp.mean(
            (pred_batch - target_flux[None, :]) ** 2,
            axis=1,
        )

        if config.use_separation_penalty:
            penalties = jax.vmap(
                lambda p: spot_separation_penalty(p, config)
            )(raw_params_batch)

            losses = losses + config.separation_penalty_weight * penalties

        return pred_batch, losses.astype(jnp.float32)

    return score_batch, predict_and_score_batch


def run_strategy_for_target(
    strategy: str,
    config: FitConfig,
    score_batch_reusable,
    predict_and_score_batch_reusable,
    target_flux: np.ndarray,
    inclination_deg: float,
):
    """
    Wrap reusable objective functions so existing strategies can stay unchanged.

    Existing strategies call:
        score_batch(raw_params_batch)

    Internally these wrappers call:
        score_batch_reusable(raw_params_batch, target_flux, inclination_deg)
    """
    target_flux_jax = jnp.asarray(target_flux, dtype=jnp.float32)
    inclination_jax = jnp.asarray(inclination_deg, dtype=jnp.float32)

    def score_batch(raw_params_batch):
        return score_batch_reusable(
            raw_params_batch,
            target_flux_jax,
            inclination_jax,
        )

    def predict_and_score_batch(raw_params_batch):
        return predict_and_score_batch_reusable(
            raw_params_batch,
            target_flux_jax,
            inclination_jax,
        )

    if strategy == "random_search":
        return run_random_search(
            config=config,
            score_batch=score_batch,
            predict_and_score_batch=predict_and_score_batch,
        )

    if strategy == "elite_mutation":
        return run_elite_mutation(
            config=config,
            score_batch=score_batch,
            predict_and_score_batch=predict_and_score_batch,
        )

    raise ValueError(f"Unknown strategy: {strategy}")


def fit_one_assumed_spot_count_for_one_example(
    target_flux: np.ndarray,
    theta_deg: np.ndarray,
    inclination_deg: float,
    assumed_n_spots: int,
    base_config: FitConfig,
    score_batch_reusable,
    predict_and_score_batch_reusable,
    args: argparse.Namespace,
    seed: int,
    save_dir: Path | None = None,
) -> dict:
    config = replace(base_config, seed=seed)

    result = run_strategy_for_target(
        strategy=args.strategy,
        config=config,
        score_batch_reusable=score_batch_reusable,
        predict_and_score_batch_reusable=predict_and_score_batch_reusable,
        target_flux=target_flux,
        inclination_deg=inclination_deg,
    )

    best_raw = np.asarray(result.best_raw_params)
    best_flux = np.asarray(result.best_flux)
    elite_raw = np.asarray(result.elite_raw_params)
    elite_losses = np.asarray(result.elite_losses)
    loss_history = np.asarray(result.loss_history)

    raw_dim = n_raw_params(config)

    mse = float(np.mean((best_flux - target_flux) ** 2))
    aic, bic = compute_information_scores(
        mse=mse,
        n_params=raw_dim,
        n_points=len(target_flux),
    )

    row = {
        "assumed_n_spots": int(assumed_n_spots),
        "mse": mse,
        "best_loss": float(result.best_loss),
        "aic": aic,
        "bic": bic,
        "n_params": int(raw_dim),
    }

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

        physical_best = physical_params_to_numpy_dict(
            best_raw,
            config=config,
        )
        physical_best = {k: np.asarray(v) for k, v in physical_best.items()}

        np.savez(
            save_dir / "result.npz",
            target_flux=target_flux,
            theta_deg=theta_deg,
            best_flux=best_flux,
            best_raw_params=best_raw,
            elite_raw_params=elite_raw,
            elite_losses=elite_losses,
            loss_history=loss_history,
            runtime_inclination_deg=np.asarray(inclination_deg, dtype=np.float32),
            **physical_best,
        )

        with (save_dir / "summary.json").open("w") as f:
            json.dump(
                {
                    "assumed_n_spots": assumed_n_spots,
                    "strategy": args.strategy,
                    "mse": mse,
                    "best_loss": float(result.best_loss),
                    "aic": aic,
                    "bic": bic,
                    "n_params": int(raw_dim),
                    "runtime_inclination_deg": float(inclination_deg),
                    "config": config_to_jsonable(config),
                },
                f,
                indent=2,
            )

    return row


def make_confusion_matrix(
    rows: list[dict],
    metric_key: str,
    n_classes: int = 5,
) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
    pred_key = f"selected_by_{metric_key}"

    for row in rows:
        true_idx = int(row["true_n_spots"]) - 1
        pred_idx = int(row[pred_key]) - 1
        matrix[true_idx, pred_idx] += 1

    return matrix


def plot_confusion_matrix(
    matrix: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", aspect="equal")
    plt.title(title)
    plt.xlabel("Selected number of spots")
    plt.ylabel("True number of spots")
    plt.xticks(np.arange(5), [1, 2, 3, 4, 5])
    plt.yticks(np.arange(5), [1, 2, 3, 4, 5])
    plt.colorbar()

    max_val = matrix.max() if matrix.size else 0
    threshold = max_val / 2.0 if max_val > 0 else 0.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            color = "white" if val > threshold else "black"
            plt.text(j, i, str(val), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_metric_curves_for_examples(
    per_example_rows: list[dict],
    output_path: Path,
) -> None:
    plt.figure(figsize=(9, 6))

    for row in per_example_rows:
        n_vals = np.arange(1, 6)
        valid_n = [n for n in n_vals if f"mse_{n}spot" in row]
        mse_vals = np.array(
            [row[f"mse_{n}spot"] for n in valid_n],
            dtype=np.float64,
        )

        plt.plot(
            valid_n,
            mse_vals,
            marker="o",
            alpha=0.35,
            linewidth=1,
        )

    plt.yscale("log")
    plt.xlabel("Assumed number of spots")
    plt.ylabel("Best MSE")
    plt.title("Per-example MSE curves")
    plt.xticks([1, 2, 3, 4, 5])
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch spot-count model comparison over many synthetic examples."
    )

    parser.add_argument(
        "--dataset-file",
        type=str,
        required=True,
        help="Path to dataset_*.npz",
    )

    parser.add_argument(
        "--flux-key",
        type=str,
        default="flux",
        choices=["flux", "flux_noisy"],
    )

    parser.add_argument(
        "--examples-per-class",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--selection",
        type=str,
        default="evenly_spaced",
        choices=["evenly_spaced", "random"],
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="elite_mutation",
        choices=["random_search", "elite_mutation"],
    )

    parser.add_argument("--min-spots", type=int, default=1)
    parser.add_argument("--max-spots", type=int, default=5)

    parser.add_argument("--fit-inclination", action="store_true")
    parser.add_argument("--use-true-inclination", action="store_true")
    parser.add_argument("--inclination", type=float, default=None)

    parser.add_argument("--n-candidates", type=int, default=2048)
    parser.add_argument("--n-elites", type=int, default=128)
    parser.add_argument("--n-steps", type=int, default=75)

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

    parser.add_argument(
        "--save-individual-results",
        action="store_true",
        help="Save result.npz and summary.json for every example and assumed spot count. Uses much more disk.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.min_spots < 1 or args.max_spots > 5 or args.min_spots > args.max_spots:
        raise ValueError("Require 1 <= min_spots <= max_spots <= 5")

    if args.fit_inclination and args.use_true_inclination:
        raise ValueError("Use either --fit-inclination or --use-true-inclination, not both.")

    if not args.fit_inclination and not args.use_true_inclination and args.inclination is None:
        raise ValueError(
            "Provide one of --fit-inclination, --use-true-inclination, or --inclination VALUE."
        )

    dataset_file = Path(args.dataset_file).expanduser().resolve()

    if not dataset_file.exists():
        raise FileNotFoundError(f"Could not find dataset file: {dataset_file}")

    timestamp = make_timestamp()
    output_dir = Path(args.output_dir) / f"batch_compare_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("JAX devices:", jax.devices(), flush=True)
    print("JAX backend:", jax.default_backend(), flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    example_specs = choose_example_indices(
        dataset_file=dataset_file,
        examples_per_class=args.examples_per_class,
        seed=args.seed,
        selection=args.selection,
    )

    print(f"Testing {len(example_specs)} examples.", flush=True)

    examples = []

    for example_number, spec in enumerate(example_specs, start=1):
        index = int(spec["index"])

        example = load_single_dataset_example(
            dataset_file=dataset_file,
            index=index,
            flux_key=args.flux_key,
        )

        if args.fit_inclination:
            runtime_inclination = 0.0
        elif args.use_true_inclination:
            runtime_inclination = float(example["inclination"])
        else:
            runtime_inclination = float(args.inclination)

        examples.append(
            {
                "example_number": example_number,
                "index": index,
                "class_offset": int(spec["class_offset"]),
                "true_n_spots": int(spec["true_n_spots"]),
                "true_inclination": float(example["inclination"]),
                "runtime_inclination": float(runtime_inclination),
                "target_flux": example["flux"].astype(np.float32),
                "theta_deg": example["theta_deg"].astype(np.float32),
            }
        )

    theta_deg = examples[0]["theta_deg"]
    datapoints = int(len(theta_deg))

    per_fit_rows = []

    fit_results: dict[int, dict[int, dict]] = {
        int(example["index"]): {}
        for example in examples
    }

    for assumed_n_spots in range(args.min_spots, args.max_spots + 1):
        print("=" * 80, flush=True)
        print(f"Compiling/reusing functions for assumed_n_spots={assumed_n_spots}", flush=True)

        # fixed_inclination_deg is only a placeholder now. The actual inclination
        # is passed dynamically to the simulator/objective. Keeping a valid value
        # prevents helper functions like spot_separation_penalty from complaining.
        placeholder_inclination = 45.0

        base_config = FitConfig(
            n_spots=assumed_n_spots,
            datapoints=datapoints,
            fit_inclination=args.fit_inclination,
            fixed_inclination_deg=None if args.fit_inclination else placeholder_inclination,
            bounds=ParameterBounds(),
            raw_init_scale=args.raw_init_scale,
            raw_mutation_scale=args.raw_mutation_scale,
            use_separation_penalty=not args.no_separation_penalty,
            min_spot_gap_deg=args.min_spot_gap_deg,
            separation_penalty_weight=args.separation_penalty_weight,
            n_candidates=args.n_candidates,
            n_elites=args.n_elites,
            n_steps=args.n_steps,
            seed=args.seed + 10_000 * assumed_n_spots,
            output_dir=output_dir,
        )

        simulate_batch = make_simulator(
            config=base_config,
            theta_deg=theta_deg,
        )

        score_batch_reusable, predict_and_score_batch_reusable = make_reusable_objective(
            config=base_config,
            simulate_batch=simulate_batch,
        )

        raw_dim = n_raw_params(base_config)

        print("[warmup] compiling objective for this assumed spot count...", flush=True)

        test_raw = jnp.zeros((args.n_candidates, raw_dim), dtype=jnp.float32)
        test_target = jnp.asarray(examples[0]["target_flux"], dtype=jnp.float32)
        test_inclination = jnp.asarray(examples[0]["runtime_inclination"], dtype=jnp.float32)

        test_losses = score_batch_reusable(
            test_raw,
            test_target,
            test_inclination,
        )
        test_losses.block_until_ready()

        test_pred, test_losses_2 = predict_and_score_batch_reusable(
            test_raw[:1],
            test_target,
            test_inclination,
        )
        test_pred.block_until_ready()
        test_losses_2.block_until_ready()

        print("[warmup] done.", flush=True)

        for example in examples:
            example_number = int(example["example_number"])
            index = int(example["index"])
            true_n_spots = int(example["true_n_spots"])

            print(
                f"  example {example_number}/{len(examples)} | "
                f"index={index} | true={true_n_spots} | assumed={assumed_n_spots}",
                flush=True,
            )

            active_config = replace(
                base_config,
                seed=args.seed + 1_000_000 * example_number + 10_000 * assumed_n_spots,
            )

            save_dir = None
            if args.save_individual_results:
                save_dir = (
                    output_dir
                    / "individual_results"
                    / f"example_{index}"
                    / f"{assumed_n_spots}spot"
                )

            fit_row = fit_one_assumed_spot_count_for_one_example(
                target_flux=example["target_flux"],
                theta_deg=theta_deg,
                inclination_deg=float(example["runtime_inclination"]),
                assumed_n_spots=assumed_n_spots,
                base_config=active_config,
                score_batch_reusable=score_batch_reusable,
                predict_and_score_batch_reusable=predict_and_score_batch_reusable,
                args=args,
                seed=active_config.seed,
                save_dir=save_dir,
            )

            fit_row.update(
                {
                    "example_number": example_number,
                    "index": index,
                    "class_offset": int(example["class_offset"]),
                    "true_n_spots": true_n_spots,
                    "true_inclination": float(example["true_inclination"]),
                    "runtime_inclination": float(example["runtime_inclination"]),
                    "flux_key": args.flux_key,
                }
            )

            fit_results[index][assumed_n_spots] = fit_row
            per_fit_rows.append(fit_row)

            print(
                f"    mse={fit_row['mse']:.3e} "
                f"aic={fit_row['aic']:.3f} "
                f"bic={fit_row['bic']:.3f}",
                flush=True,
            )

            write_csv(per_fit_rows, output_dir / "per_fit_results_partial.csv")

    per_example_rows = []

    for example in examples:
        index = int(example["index"])
        true_n_spots = int(example["true_n_spots"])

        fit_rows = [
            fit_results[index][assumed_n_spots]
            for assumed_n_spots in range(args.min_spots, args.max_spots + 1)
        ]

        selected_by_mse = min(fit_rows, key=lambda r: r["mse"])["assumed_n_spots"]
        selected_by_aic = min(fit_rows, key=lambda r: r["aic"])["assumed_n_spots"]
        selected_by_bic = min(fit_rows, key=lambda r: r["bic"])["assumed_n_spots"]

        example_row = {
            "example_number": int(example["example_number"]),
            "index": index,
            "class_offset": int(example["class_offset"]),
            "true_n_spots": true_n_spots,
            "true_inclination": float(example["true_inclination"]),
            "selected_by_mse": int(selected_by_mse),
            "selected_by_aic": int(selected_by_aic),
            "selected_by_bic": int(selected_by_bic),
            "correct_by_mse": int(selected_by_mse == true_n_spots),
            "correct_by_aic": int(selected_by_aic == true_n_spots),
            "correct_by_bic": int(selected_by_bic == true_n_spots),
        }

        for row in fit_rows:
            n = row["assumed_n_spots"]
            example_row[f"mse_{n}spot"] = row["mse"]
            example_row[f"aic_{n}spot"] = row["aic"]
            example_row[f"bic_{n}spot"] = row["bic"]

        per_example_rows.append(example_row)

    write_csv(per_fit_rows, output_dir / "per_fit_results.csv")
    write_csv(per_example_rows, output_dir / "per_example_results.csv")

    confusion_mse = make_confusion_matrix(per_example_rows, "mse")
    confusion_aic = make_confusion_matrix(per_example_rows, "aic")
    confusion_bic = make_confusion_matrix(per_example_rows, "bic")

    np.savez(
        output_dir / "confusion_matrices.npz",
        confusion_mse=confusion_mse,
        confusion_aic=confusion_aic,
        confusion_bic=confusion_bic,
    )

    plot_confusion_matrix(
        confusion_mse,
        "Selected spot count by MSE",
        output_dir / "confusion_mse.png",
    )

    plot_confusion_matrix(
        confusion_aic,
        "Selected spot count by AIC-like score",
        output_dir / "confusion_aic.png",
    )

    plot_confusion_matrix(
        confusion_bic,
        "Selected spot count by BIC-like score",
        output_dir / "confusion_bic.png",
    )

    plot_metric_curves_for_examples(
        per_example_rows,
        output_dir / "mse_curves.png",
    )

    accuracy_mse = float(np.mean([row["correct_by_mse"] for row in per_example_rows]))
    accuracy_aic = float(np.mean([row["correct_by_aic"] for row in per_example_rows]))
    accuracy_bic = float(np.mean([row["correct_by_bic"] for row in per_example_rows]))

    summary = {
        "dataset_file": str(dataset_file),
        "flux_key": args.flux_key,
        "strategy": args.strategy,
        "examples_per_class": args.examples_per_class,
        "selection": args.selection,
        "min_spots": args.min_spots,
        "max_spots": args.max_spots,
        "n_candidates": args.n_candidates,
        "n_elites": args.n_elites,
        "n_steps": args.n_steps,
        "fit_inclination": args.fit_inclination,
        "use_true_inclination": args.use_true_inclination,
        "fixed_inclination": args.inclination,
        "accuracy_mse": accuracy_mse,
        "accuracy_aic": accuracy_aic,
        "accuracy_bic": accuracy_bic,
        "confusion_mse": confusion_mse.tolist(),
        "confusion_aic": confusion_aic.tolist(),
        "confusion_bic": confusion_bic.tolist(),
    }

    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print("\nBatch comparison complete.", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Accuracy by MSE: {accuracy_mse:.4f}", flush=True)
    print(f"Accuracy by AIC: {accuracy_aic:.4f}", flush=True)
    print(f"Accuracy by BIC: {accuracy_bic:.4f}", flush=True)


if __name__ == "__main__":
    main()