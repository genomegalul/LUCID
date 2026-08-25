from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import jax
import jax.numpy as jnp
from jax import jit, vjp

from jaxoplanet.starry.ylm import Ylm, ylm_spot
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.light_curves import surface_light_curve

from data_split import make_datasets, summarize_split
from model import RetrievalNet, count_parameters


# --- Default configuration ---

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

WEIGHTS_DIR = SCRIPT_DIR / "weights"
RUNS_DIR = SCRIPT_DIR / "runs"
PLOTS_DIR = SCRIPT_DIR / "plots"

BATCH_SIZE = 64
NUM_EPOCHS = 120
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 0
RANDOM_SEED = 0

CHUNK_FOR_JAX = 16

YDEG = 11
YSIZE = (YDEG + 1) * (YDEG + 1)

SPOT_RADIUS_RANGE = (4.0, 18.0)
SPOT_CONTRAST_RANGE = (0.3, 0.8)
SQUASH_SCALE = 0.3

LR_SCHED_FACTOR = 0.5
LR_SCHED_PATIENCE = 3

N_MC_UNCERTAINTY = 100


# --- Global JAX simulator handles ---

THETA_DEG = None
UNIT_DENSE = None
SPOT_FN = None
LC_SMALLBATCH = None
JT_G_SMALLBATCH = None
SIM_N_SPOTS = None
SIM_DATAPOINTS = None


# --- Utilities ---

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch, device: torch.device):
    (lc_in, aux), target_lc, metadata = batch
    lc_in = lc_in.to(device=device, dtype=torch.float32)
    aux = aux.to(device=device, dtype=torch.float32)
    target_lc = target_lc.to(device=device, dtype=torch.float32)
    return lc_in, aux, target_lc, metadata


# --- JAX simulator setup ---

def setup_jax_simulator(
    n_spots: int,
    datapoints: int,
) -> None:
    """
    Build JAX functions for a fixed n_spots and datapoints.

    This mirrors the synthesis simulator, but the network predicts:
        lats, lons, radii, shared contrast

    rather than reading them from files.
    """
    global THETA_DEG, UNIT_DENSE, SPOT_FN
    global LC_SMALLBATCH, JT_G_SMALLBATCH
    global SIM_N_SPOTS, SIM_DATAPOINTS

    SIM_N_SPOTS = int(n_spots)
    SIM_DATAPOINTS = int(datapoints)

    THETA_DEG = jnp.linspace(
        0.0,
        360.0,
        SIM_DATAPOINTS,
        endpoint=False,
        dtype=jnp.float32,
    )

    UNIT_DENSE = jnp.zeros(YSIZE, dtype=jnp.float32).at[0].set(1.0)

    print("Forcing 'ylm_spot' initialization to CPU to bypass potential cuSOLVER issues...")
    cpu_device = jax.devices("cpu")[0]
    with jax.default_device(cpu_device):
        SPOT_FN = ylm_spot(YDEG)
    print("... 'ylm_spot' initialized successfully on CPU.")

    def squash_raw_params(raw_params: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        spot_raw = raw_params[: SIM_N_SPOTS * 3].reshape((SIM_N_SPOTS, 3))
        contrast_raw = raw_params[-1]

        lats = 89.5 * jnp.tanh(SQUASH_SCALE * spot_raw[:, 0])
        lons = 180.0 * jnp.tanh(SQUASH_SCALE * spot_raw[:, 1])

        rmin, rmax = SPOT_RADIUS_RANGE
        radii = (rmin + 1e-3) + (rmax - (rmin + 1e-3)) * jax.nn.sigmoid(
            SQUASH_SCALE * spot_raw[:, 2]
        )

        cmin, cmax = SPOT_CONTRAST_RANGE
        contrast = cmin + (cmax - cmin) * jax.nn.sigmoid(
            SQUASH_SCALE * contrast_raw
        )

        return lats, lons, radii, contrast

    def build_spot_map(
        lats: jnp.ndarray,
        lons: jnp.ndarray,
        radii: jnp.ndarray,
        contrast: jnp.ndarray,
    ) -> jnp.ndarray:
        def add_one(i, y_curr):
            spot = SPOT_FN(
                contrast=contrast,
                r=jnp.deg2rad(radii[i]),
                lat=jnp.deg2rad(lats[i]),
                lon=jnp.deg2rad(lons[i]),
            )
            y_spot = spot.todense()
            return y_curr + (y_spot - UNIT_DENSE)

        return jax.lax.fori_loop(0, SIM_N_SPOTS, add_one, UNIT_DENSE)

    @jit
    def flux_from_map_ylm(
        y_dense: jnp.ndarray,
        inc_deg: jnp.ndarray,
    ) -> jnp.ndarray:
        surface = Surface(
            y=Ylm.from_dense(y_dense, normalize=False),
            inc=jnp.deg2rad(inc_deg),
        )

        def step(theta):
            return surface_light_curve(surface, theta=jnp.deg2rad(theta))

        return jax.vmap(step)(THETA_DEG).astype(jnp.float32)

    def lc_single_impl(
        raw_params: jnp.ndarray,
        aux: jnp.ndarray,
    ) -> jnp.ndarray:
        lats, lons, radii, contrast = squash_raw_params(raw_params)

        sin_i, cos_i = aux
        inc_deg = jnp.rad2deg(jnp.arctan2(sin_i, cos_i))

        y_dense = build_spot_map(
            lats=lats,
            lons=lons,
            radii=radii,
            contrast=contrast,
        )

        flux = flux_from_map_ylm(
            y_dense=y_dense,
            inc_deg=inc_deg,
        )

        return jnp.nan_to_num(flux, nan=1.0, posinf=1.0, neginf=1.0)

    def jt_g_single_impl(
        raw_params: jnp.ndarray,
        aux: jnp.ndarray,
        g: jnp.ndarray,
    ) -> jnp.ndarray:
        def fun(p):
            return lc_single_impl(p, aux)

        _, vjp_fun = vjp(fun, raw_params)
        grad = vjp_fun(g)[0]
        return jnp.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

    def lc_smallbatch_impl(
        raw_batch: jnp.ndarray,
        aux_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(lc_single_impl, in_axes=(0, 0))(raw_batch, aux_batch)

    def jt_g_smallbatch_impl(
        raw_batch: jnp.ndarray,
        aux_batch: jnp.ndarray,
        g_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        return jax.vmap(jt_g_single_impl, in_axes=(0, 0, 0))(
            raw_batch,
            aux_batch,
            g_batch,
        )

    LC_SMALLBATCH = jit(lc_smallbatch_impl)
    JT_G_SMALLBATCH = jit(jt_g_smallbatch_impl)


class JaxLightCurveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw_params: torch.Tensor, aux: torch.Tensor):
        if LC_SMALLBATCH is None:
            raise RuntimeError("JAX simulator has not been initialized. Call setup_jax_simulator first.")

        p_np = raw_params.detach().cpu().numpy()
        a_np = aux.detach().cpu().numpy()

        batch_size = p_np.shape[0]
        outs = []

        for start in range(0, batch_size, CHUNK_FOR_JAX):
            end = min(start + CHUNK_FOR_JAX, batch_size)

            out_chunk = np.asarray(
                LC_SMALLBATCH(
                    jnp.asarray(p_np[start:end], dtype=jnp.float32),
                    jnp.asarray(a_np[start:end], dtype=jnp.float32),
                )
            )

            outs.append(np.nan_to_num(out_chunk, nan=1.0, posinf=1.0, neginf=1.0))

        out = torch.from_numpy(np.vstack(outs)).to(raw_params.device)
        ctx.save_for_backward(raw_params.detach().cpu(), aux.detach().cpu())
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if JT_G_SMALLBATCH is None:
            raise RuntimeError("JAX simulator has not been initialized. Call setup_jax_simulator first.")

        raw_cpu, aux_cpu = ctx.saved_tensors

        p_np = raw_cpu.numpy()
        a_np = aux_cpu.numpy()
        g_np = torch.nan_to_num(
            grad_output,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).detach().cpu().numpy()

        batch_size = p_np.shape[0]
        grads = []

        for start in range(0, batch_size, CHUNK_FOR_JAX):
            end = min(start + CHUNK_FOR_JAX, batch_size)

            grad_chunk = np.asarray(
                JT_G_SMALLBATCH(
                    jnp.asarray(p_np[start:end], dtype=jnp.float32),
                    jnp.asarray(a_np[start:end], dtype=jnp.float32),
                    jnp.asarray(g_np[start:end], dtype=jnp.float32),
                )
            )

            grads.append(np.nan_to_num(grad_chunk, nan=0.0, posinf=0.0, neginf=0.0))

        grad_params = torch.from_numpy(np.vstack(grads)).to(grad_output.device)
        return grad_params, None


def simulate_lc_from_params_torch(
    raw_params: torch.Tensor,
    aux: torch.Tensor,
) -> torch.Tensor:
    return JaxLightCurveFunction.apply(raw_params, aux)


# --- Training / evaluation ---

def run_epoch(
    model: RetrievalNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    sample_model: bool,
    grad_clip_norm: float | None = None,
) -> dict:
    is_train = optimizer is not None
    model.train(mode=is_train)

    total_mse = 0.0
    total_count = 0

    for batch in loader:
        lc_in, aux, target_lc, _metadata = move_batch_to_device(batch, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        raw = model(lc_in, aux, sample=sample_model)
        pred_lc = simulate_lc_from_params_torch(raw.float(), aux)

        mse = F.mse_loss(pred_lc, target_lc, reduction="mean")

        if is_train:
            mse.backward()

            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

            optimizer.step()

        batch_size = target_lc.shape[0]
        total_mse += float(mse.item()) * batch_size
        total_count += batch_size

    return {
        "mse": total_mse / max(total_count, 1),
    }


def save_history_csv(history: list[dict], path: Path) -> None:
    fieldnames = ["epoch", "train_mse", "val_mse", "test_mse", "learning_rate"]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def plot_history(history: list[dict], path: Path, title: str) -> None:
    epochs = np.asarray([row["epoch"] for row in history], dtype=np.int32)
    train = np.asarray([row["train_mse"] for row in history], dtype=np.float64)
    val = np.asarray([row["val_mse"] for row in history], dtype=np.float64)
    test = np.asarray([row["test_mse"] for row in history], dtype=np.float64)

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train, label="Train MSE")
    plt.plot(epochs, val, label="Val MSE")
    plt.plot(epochs, test, label="Test MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def save_uncertainty_example(
    model: RetrievalNet,
    loader: DataLoader,
    device: torch.device,
    n_samples: int,
    output_npz: Path,
    output_png: Path,
) -> None:
    model.eval()

    batch = next(iter(loader))
    lc_in, aux, target_lc, metadata = move_batch_to_device(batch, device)

    lc_in = lc_in[:1]
    aux = aux[:1]
    target = target_lc[:1]

    preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            raw = model(lc_in, aux, sample=True)
            pred = simulate_lc_from_params_torch(raw.float(), aux)
            preds.append(pred.squeeze(0).cpu().numpy())

    preds = np.asarray(preds, dtype=np.float32)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    target_np = target.squeeze(0).cpu().numpy()

    np.savez(
        output_npz,
        target=target_np,
        mean=mean,
        std=std,
        samples=preds,
    )

    x = np.arange(target_np.shape[0])

    plt.figure(figsize=(10, 5))
    plt.plot(x, target_np, color="black", lw=2, label="Target")
    plt.plot(x, mean, lw=2, label="Mean prediction")
    plt.fill_between(
        x,
        mean - 2.0 * std,
        mean + 2.0 * std,
        alpha=0.3,
        label="±2 std",
    )
    plt.xlabel("Phase index")
    plt.ylabel("Flux")
    plt.title("Retrieval uncertainty example")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close()


def train_one_model(args: argparse.Namespace, n_spots: int) -> None:
    ensure_dirs()
    set_seed(args.seed)

    device = get_device()
    timestamp = make_timestamp()

    print("=" * 80, flush=True)
    print(f"Training retrieval model for n_spots={n_spots}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"JAX backend: {jax.default_backend()}", flush=True)
    print("=" * 80, flush=True)

    split_summary = summarize_split(
        n_spots=n_spots,
        data_dir=args.data_dir,
        datapoints=args.datapoints,
        dataset_file=args.dataset_file,
        flux_key=args.flux_key,
    )

    print("Split summary:", json.dumps(split_summary, indent=2), flush=True)

    train_dataset, val_dataset, test_dataset, store = make_datasets(
        n_spots=n_spots,
        data_dir=args.data_dir,
        datapoints=args.datapoints,
        dataset_file=args.dataset_file,
        flux_key=args.flux_key,
        normalize_input=True,
    )

    datapoints = store.datapoints

    setup_jax_simulator(
        n_spots=n_spots,
        datapoints=datapoints,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = RetrievalNet(
        n_spots=n_spots,
        input_length=datapoints,
        dropout_p=args.dropout,
        rho_init=args.rho_init,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_sched_factor,
        patience=args.lr_sched_patience,
        min_lr=args.min_lr,
    )

    best_val_mse = float("inf")
    best_epoch = None

    weights_path = WEIGHTS_DIR / f"{n_spots}spot_weights_{timestamp}.pt"
    final_weights_path = WEIGHTS_DIR / f"{n_spots}spot_weights_final_{timestamp}.pt"

    history: list[dict] = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            sample_model=True,
            grad_clip_norm=args.grad_clip_norm,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                device=device,
                sample_model=False,
            )

            test_metrics = run_epoch(
                model=model,
                loader=test_loader,
                optimizer=None,
                device=device,
                sample_model=False,
            )

        scheduler.step(val_metrics["mse"])

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_mse": train_metrics["mse"],
            "val_mse": val_metrics["mse"],
            "test_mse": test_metrics["mse"],
            "learning_rate": current_lr,
        }
        history.append(row)

        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d}  "
            f"train_mse={train_metrics['mse']:.8e}  "
            f"val_mse={val_metrics['mse']:.8e}  "
            f"test_mse={test_metrics['mse']:.8e}  "
            f"lr={current_lr:.3e}  "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch

            checkpoint = {
                "timestamp": timestamp,
                "epoch": epoch,
                "n_spots": n_spots,
                "datapoints": datapoints,
                "model_name": "RetrievalNet",
                "model_kwargs": {
                    "n_spots": n_spots,
                    "input_length": datapoints,
                    "dropout_p": args.dropout,
                    "rho_init": args.rho_init,
                },
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_mse": best_val_mse,
                "train_config": vars(args),
                "split_summary": split_summary,
            }

            torch.save(checkpoint, weights_path)
            print(f"Saved best checkpoint to: {weights_path}", flush=True)

    total_time = time.time() - start_time

    torch.save(
        {
            "timestamp": timestamp,
            "epoch": args.epochs,
            "n_spots": n_spots,
            "datapoints": datapoints,
            "model_name": "RetrievalNet",
            "model_kwargs": {
                "n_spots": n_spots,
                "input_length": datapoints,
                "dropout_p": args.dropout,
                "rho_init": args.rho_init,
            },
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_config": vars(args),
            "split_summary": split_summary,
        },
        final_weights_path,
    )

    history_csv = RUNS_DIR / f"{n_spots}spot_history_{timestamp}.csv"
    save_history_csv(history, history_csv)

    history_plot = PLOTS_DIR / f"{n_spots}spot_loss_curve_{timestamp}.png"
    plot_history(
        history=history,
        path=history_plot,
        title=f"{n_spots}-spot retrieval loss",
    )

    summary_path = RUNS_DIR / f"{n_spots}spot_summary_{timestamp}.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "n_spots": n_spots,
                "datapoints": datapoints,
                "best_epoch": best_epoch,
                "best_val_mse": best_val_mse,
                "total_training_time_sec": total_time,
                "weights_path": str(weights_path),
                "final_weights_path": str(final_weights_path),
                "history_csv": str(history_csv),
                "history_plot": str(history_plot),
                "train_config": vars(args),
                "split_summary": split_summary,
            },
            f,
            indent=2,
        )

    print("Loading best model for uncertainty example...", flush=True)
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    uncertainty_npz = RUNS_DIR / f"{n_spots}spot_uncertainty_{timestamp}.npz"
    uncertainty_png = PLOTS_DIR / f"{n_spots}spot_uncertainty_{timestamp}.png"

    save_uncertainty_example(
        model=model,
        loader=val_loader,
        device=device,
        n_samples=args.mc_samples,
        output_npz=uncertainty_npz,
        output_png=uncertainty_png,
    )

    print("\nTraining complete.", flush=True)
    print(f"Best checkpoint:      {weights_path}", flush=True)
    print(f"Final checkpoint:     {final_weights_path}", flush=True)
    print(f"History CSV:          {history_csv}", flush=True)
    print(f"Loss plot:            {history_plot}", flush=True)
    print(f"Summary JSON:         {summary_path}", flush=True)
    print(f"Uncertainty NPZ:      {uncertainty_npz}", flush=True)
    print(f"Uncertainty plot:     {uncertainty_png}", flush=True)

    store.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train retrieval models that infer spot parameters by differentiating through a JAX light-curve simulator."
    )

    parser.add_argument("--n-spots", type=int, default=None, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--all", action="store_true", help="Train all five retrieval models sequentially.")

    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--dataset-file", type=str, default=None)
    parser.add_argument("--datapoints", type=int, default=256)
    parser.add_argument("--flux-key", type=str, default="flux", choices=["flux", "flux_noisy"])

    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--grad-clip-norm", type=float, default=GRAD_CLIP_NORM)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)

    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--rho-init", type=float, default=-5.0)

    parser.add_argument("--lr-sched-factor", type=float, default=LR_SCHED_FACTOR)
    parser.add_argument("--lr-sched-patience", type=int, default=LR_SCHED_PATIENCE)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--mc-samples", type=int, default=N_MC_UNCERTAINTY)

    args = parser.parse_args()

    if args.all and args.n_spots is not None:
        raise ValueError("Use either --all or --n-spots, not both.")

    if not args.all and args.n_spots is None:
        raise ValueError("Provide either --all or --n-spots.")

    return args


def main() -> None:
    args = parse_args()

    if args.all:
        for n_spots in [1, 2, 3, 4, 5]:
            train_one_model(args, n_spots=n_spots)
    else:
        train_one_model(args, n_spots=args.n_spots)


if __name__ == "__main__":
    main()