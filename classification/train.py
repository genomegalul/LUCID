from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_split import make_datasets, summarize_splits
from model import SpotCountCNN


# --- Configuration ---

SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SCRIPT_DIR / "weights"
RUNS_DIR = SCRIPT_DIR / "runs"
PLOTS_DIR = SCRIPT_DIR / "plots"

DATA_DIR = SCRIPT_DIR.parent / "data"

DATAPOINTS = 256
FLUX_KEY = "flux"

BATCH_SIZE = 512
NUM_EPOCHS = 120
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 0
NUM_WORKERS = 4

NORMALIZE_INPUT = True
SAVE_BEST_BY = "val_accuracy"   # "val_loss" or "val_accuracy"

# Multitask loss:
#   total_loss = CE + LAMBDA_COUNT * count_loss
LAMBDA_COUNT = 0.10


# --- Helpers ---

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dirs() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def move_batch_to_device(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device=device, dtype=torch.float32)
    inc = batch["inclination"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device, dtype=torch.long)
    return x, inc, y


def target_count_from_labels(y: torch.Tensor) -> torch.Tensor:
    """
    Convert class index labels [0, 4] to physical spot counts [1, 5].
    """
    return y.to(dtype=torch.float32) + 1.0


def compute_multitask_loss(
    logits: torch.Tensor,
    count_pred: torch.Tensor,
    y: torch.Tensor,
    ce_criterion: nn.Module,
    count_criterion: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce_loss = ce_criterion(logits, y)
    count_target = target_count_from_labels(y)
    count_loss = count_criterion(count_pred, count_target)
    total_loss = ce_loss + LAMBDA_COUNT * count_loss
    return total_loss, ce_loss, count_loss


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    ce_criterion: nn.Module,
    count_criterion: nn.Module,
    device: torch.device,
    collect_outputs: bool = False,
) -> dict:
    model.eval()

    running_loss = 0.0
    running_ce_loss = 0.0
    running_count_loss = 0.0
    running_count_abs_error = 0.0
    running_correct = 0
    running_total = 0

    all_probs = []
    all_preds = []
    all_targets = []
    all_count_preds = []
    all_paths = []
    all_metadata = []

    with torch.no_grad():
        for batch in loader:
            x, inc, y = move_batch_to_device(batch, device)

            logits, count_pred = model.forward_with_count(x, inc)

            loss, ce_loss, count_loss = compute_multitask_loss(
                logits=logits,
                count_pred=count_pred,
                y=y,
                ce_criterion=ce_criterion,
                count_criterion=count_criterion,
            )

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            count_target = target_count_from_labels(y)
            count_abs_error = torch.abs(count_pred - count_target)

            batch_size = y.size(0)

            running_loss += loss.item() * batch_size
            running_ce_loss += ce_loss.item() * batch_size
            running_count_loss += count_loss.item() * batch_size
            running_count_abs_error += count_abs_error.sum().item()

            running_correct += (preds == y).sum().item()
            running_total += batch_size

            if collect_outputs:
                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_targets.extend(y.cpu().tolist())
                all_count_preds.extend(count_pred.cpu().tolist())

                if "path" in batch:
                    all_paths.extend(batch["path"])
                elif "index" in batch:
                    all_paths.extend([f"index_{int(i)}" for i in batch["index"]])
                else:
                    all_paths.extend(["unknown"] * batch_size)

                all_metadata.extend(
                    [
                        {
                            "n_spots": int(batch["n_spots"][i]),
                            "inclination": float(batch["inclination"][i]),
                            "contrast": float(batch["contrast"][i]),
                            "noise": float(batch["noise"][i]),
                            "count_pred": float(count_pred.cpu()[i]),
                        }
                        for i in range(batch_size)
                    ]
                )

    avg_loss = running_loss / max(running_total, 1)
    avg_ce_loss = running_ce_loss / max(running_total, 1)
    avg_count_loss = running_count_loss / max(running_total, 1)
    avg_count_mae = running_count_abs_error / max(running_total, 1)
    avg_acc = running_correct / max(running_total, 1)

    result = {
        "loss": avg_loss,
        "ce_loss": avg_ce_loss,
        "count_loss": avg_count_loss,
        "count_mae": avg_count_mae,
        "accuracy": avg_acc,
    }

    if collect_outputs:
        result["probabilities"] = all_probs
        result["predictions"] = all_preds
        result["targets"] = all_targets
        result["count_predictions"] = all_count_preds
        result["paths"] = all_paths
        result["metadata"] = all_metadata

    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    ce_criterion: nn.Module,
    count_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    model.train()

    running_loss = 0.0
    running_ce_loss = 0.0
    running_count_loss = 0.0
    running_count_abs_error = 0.0
    running_correct = 0
    running_total = 0

    for batch in loader:
        x, inc, y = move_batch_to_device(batch, device)

        optimizer.zero_grad()

        logits, count_pred = model.forward_with_count(x, inc)

        loss, ce_loss, count_loss = compute_multitask_loss(
            logits=logits,
            count_pred=count_pred,
            y=y,
            ce_criterion=ce_criterion,
            count_criterion=count_criterion,
        )

        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        count_target = target_count_from_labels(y)
        count_abs_error = torch.abs(count_pred.detach() - count_target)

        batch_size = y.size(0)

        running_loss += loss.item() * batch_size
        running_ce_loss += ce_loss.item() * batch_size
        running_count_loss += count_loss.item() * batch_size
        running_count_abs_error += count_abs_error.sum().item()

        running_correct += (preds == y).sum().item()
        running_total += batch_size

    avg_loss = running_loss / max(running_total, 1)
    avg_ce_loss = running_ce_loss / max(running_total, 1)
    avg_count_loss = running_count_loss / max(running_total, 1)
    avg_count_mae = running_count_abs_error / max(running_total, 1)
    avg_acc = running_correct / max(running_total, 1)

    return {
        "loss": avg_loss,
        "ce_loss": avg_ce_loss,
        "count_loss": avg_count_loss,
        "count_mae": avg_count_mae,
        "accuracy": avg_acc,
    }


def save_history_csv(history: list[dict], csv_path: Path) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "train_ce_loss",
        "train_count_loss",
        "train_count_mae",
        "train_accuracy",
        "val_loss",
        "val_ce_loss",
        "val_count_loss",
        "val_count_mae",
        "val_accuracy",
        "learning_rate",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def make_confusion_matrix(
    predictions: list[int],
    targets: list[int],
    n_classes: int = 5,
) -> list[list[int]]:
    matrix = [[0 for _ in range(n_classes)] for _ in range(n_classes)]
    for true_idx, pred_idx in zip(targets, predictions):
        matrix[true_idx][pred_idx] += 1
    return matrix


def compute_per_class_accuracy(
    predictions: list[int],
    targets: list[int],
    n_classes: int = 5,
) -> dict:
    per_class = {}

    for class_idx in range(n_classes):
        total = sum(1 for t in targets if t == class_idx)
        correct = sum(
            1
            for t, p in zip(targets, predictions)
            if t == class_idx and p == class_idx
        )
        acc = correct / total if total > 0 else 0.0
        per_class[str(class_idx + 1)] = {
            "total": total,
            "correct": correct,
            "accuracy": acc,
        }

    return per_class


# --- Main ---

def main() -> None:
    ensure_dirs()
    set_seed(RANDOM_SEED)

    timestamp = make_timestamp()
    device = get_device()

    print(f"Using device: {device}", flush=True)
    print(f"Data directory: {DATA_DIR}", flush=True)
    print(f"Datapoints: {DATAPOINTS}", flush=True)
    print(f"Flux key: {FLUX_KEY}", flush=True)
    print(f"Lambda count: {LAMBDA_COUNT}", flush=True)

    split_summary = summarize_splits(
        data_dir=DATA_DIR,
        datapoints=DATAPOINTS,
    )
    print("Split summary:", json.dumps(split_summary, indent=2), flush=True)

    train_dataset, val_dataset, test_dataset = make_datasets(
        data_dir=DATA_DIR,
        datapoints=DATAPOINTS,
        normalize=NORMALIZE_INPUT,
        flux_key=FLUX_KEY,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    model = SpotCountCNN(input_length=DATAPOINTS, n_classes=5).to(device)

    ce_criterion = nn.CrossEntropyLoss()
    count_criterion = nn.SmoothL1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        min_lr=5e-6,
    )

    history: list[dict] = []

    best_metric = None
    best_epoch = None
    best_weights_path = WEIGHTS_DIR / f"weights_{timestamp}.pt"

    train_start_time = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            ce_criterion=ce_criterion,
            count_criterion=count_criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate_split(
            model=model,
            loader=val_loader,
            ce_criterion=ce_criterion,
            count_criterion=count_criterion,
            device=device,
            collect_outputs=False,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce_loss": train_metrics["ce_loss"],
            "train_count_loss": train_metrics["count_loss"],
            "train_count_mae": train_metrics["count_mae"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_ce_loss": val_metrics["ce_loss"],
            "val_count_loss": val_metrics["count_loss"],
            "val_count_mae": val_metrics["count_mae"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": current_lr,
        }
        history.append(row)

        elapsed = time.time() - epoch_start

        print(
            f"Epoch {epoch:03d}/{NUM_EPOCHS:03d}  "
            f"train_loss={train_metrics['loss']:.6f}  "
            f"train_ce={train_metrics['ce_loss']:.6f}  "
            f"train_count={train_metrics['count_loss']:.6f}  "
            f"train_mae={train_metrics['count_mae']:.4f}  "
            f"train_acc={train_metrics['accuracy']:.4f}  "
            f"val_loss={val_metrics['loss']:.6f}  "
            f"val_ce={val_metrics['ce_loss']:.6f}  "
            f"val_count={val_metrics['count_loss']:.6f}  "
            f"val_mae={val_metrics['count_mae']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}  "
            f"lr={current_lr:.2e}  "
            f"time={elapsed:.1f}s",
            flush=True,
        )

        scheduler.step(val_metrics["accuracy"])

        if SAVE_BEST_BY == "val_loss":
            metric_value = val_metrics["loss"]
            is_better = best_metric is None or metric_value < best_metric
        elif SAVE_BEST_BY == "val_accuracy":
            metric_value = val_metrics["accuracy"]
            is_better = best_metric is None or metric_value > best_metric
        else:
            raise ValueError(f"Unsupported SAVE_BEST_BY: {SAVE_BEST_BY}")

        if is_better:
            best_metric = metric_value
            best_epoch = epoch

            checkpoint = {
                "timestamp": timestamp,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": best_metric,
                "save_best_by": SAVE_BEST_BY,
                "model_name": "SpotCountCNN",
                "model_kwargs": {
                    "input_length": DATAPOINTS,
                    "n_classes": 5,
                },
                "train_config": {
                    "batch_size": BATCH_SIZE,
                    "num_epochs": NUM_EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "random_seed": RANDOM_SEED,
                    "num_workers": NUM_WORKERS,
                    "normalize_input": NORMALIZE_INPUT,
                    "data_dir": str(DATA_DIR),
                    "datapoints": DATAPOINTS,
                    "flux_key": FLUX_KEY,
                    "lambda_count": LAMBDA_COUNT,
                    "loss": "cross_entropy + lambda_count * smooth_l1_count",
                },
                "split_summary": split_summary,
            }
            torch.save(checkpoint, best_weights_path)
            print(f"Saved best checkpoint to: {best_weights_path}", flush=True)

    total_training_time = time.time() - train_start_time

    print("Loading best checkpoint for final evaluation...", flush=True)
    checkpoint = torch.load(best_weights_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_train_metrics = evaluate_split(
        model=model,
        loader=train_loader,
        ce_criterion=ce_criterion,
        count_criterion=count_criterion,
        device=device,
        collect_outputs=False,
    )
    final_val_metrics = evaluate_split(
        model=model,
        loader=val_loader,
        ce_criterion=ce_criterion,
        count_criterion=count_criterion,
        device=device,
        collect_outputs=False,
    )
    final_test_metrics = evaluate_split(
        model=model,
        loader=test_loader,
        ce_criterion=ce_criterion,
        count_criterion=count_criterion,
        device=device,
        collect_outputs=True,
    )

    test_confusion = make_confusion_matrix(
        predictions=final_test_metrics["predictions"],
        targets=final_test_metrics["targets"],
        n_classes=5,
    )
    test_per_class_accuracy = compute_per_class_accuracy(
        predictions=final_test_metrics["predictions"],
        targets=final_test_metrics["targets"],
        n_classes=5,
    )

    history_csv_path = RUNS_DIR / f"run_{timestamp}_history.csv"
    save_history_csv(history, history_csv_path)

    summary_json_path = RUNS_DIR / f"run_{timestamp}_summary.json"
    summary_payload = {
        "timestamp": timestamp,
        "weights_path": str(best_weights_path),
        "history_csv_path": str(history_csv_path),
        "save_best_by": SAVE_BEST_BY,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "device": str(device),
        "total_training_time_sec": total_training_time,
        "train_config": {
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "random_seed": RANDOM_SEED,
            "num_workers": NUM_WORKERS,
            "normalize_input": NORMALIZE_INPUT,
            "data_dir": str(DATA_DIR),
            "datapoints": DATAPOINTS,
            "flux_key": FLUX_KEY,
            "lambda_count": LAMBDA_COUNT,
            "loss": "cross_entropy + lambda_count * smooth_l1_count",
        },
        "split_summary": split_summary,
        "final_metrics": {
            "train": {
                "loss": final_train_metrics["loss"],
                "ce_loss": final_train_metrics["ce_loss"],
                "count_loss": final_train_metrics["count_loss"],
                "count_mae": final_train_metrics["count_mae"],
                "accuracy": final_train_metrics["accuracy"],
            },
            "val": {
                "loss": final_val_metrics["loss"],
                "ce_loss": final_val_metrics["ce_loss"],
                "count_loss": final_val_metrics["count_loss"],
                "count_mae": final_val_metrics["count_mae"],
                "accuracy": final_val_metrics["accuracy"],
            },
            "test": {
                "loss": final_test_metrics["loss"],
                "ce_loss": final_test_metrics["ce_loss"],
                "count_loss": final_test_metrics["count_loss"],
                "count_mae": final_test_metrics["count_mae"],
                "accuracy": final_test_metrics["accuracy"],
            },
        },
        "test_confusion_matrix": test_confusion,
        "test_per_class_accuracy": test_per_class_accuracy,
    }

    with summary_json_path.open("w") as f:
        json.dump(summary_payload, f, indent=2)

    test_outputs_json_path = RUNS_DIR / f"run_{timestamp}_test_outputs.json"
    test_outputs_payload = {
        "timestamp": timestamp,
        "weights_path": str(best_weights_path),
        "probabilities": final_test_metrics["probabilities"],
        "predictions": [int(x) + 1 for x in final_test_metrics["predictions"]],
        "targets": [int(x) + 1 for x in final_test_metrics["targets"]],
        "count_predictions": final_test_metrics["count_predictions"],
        "paths": final_test_metrics["paths"],
        "metadata": final_test_metrics["metadata"],
    }

    with test_outputs_json_path.open("w") as f:
        json.dump(test_outputs_payload, f, indent=2)

    print("\nTraining complete.", flush=True)
    print(f"Best checkpoint: {best_weights_path}", flush=True)
    print(f"Best epoch:      {best_epoch}", flush=True)
    print(f"History CSV:     {history_csv_path}", flush=True)
    print(f"Summary JSON:    {summary_json_path}", flush=True)
    print(f"Test outputs:    {test_outputs_json_path}", flush=True)

    print("\nFinal metrics:", flush=True)
    print(
        f"Train -> loss={final_train_metrics['loss']:.6f}, "
        f"ce={final_train_metrics['ce_loss']:.6f}, "
        f"count={final_train_metrics['count_loss']:.6f}, "
        f"mae={final_train_metrics['count_mae']:.4f}, "
        f"acc={final_train_metrics['accuracy']:.4f}",
        flush=True,
    )
    print(
        f"Val   -> loss={final_val_metrics['loss']:.6f}, "
        f"ce={final_val_metrics['ce_loss']:.6f}, "
        f"count={final_val_metrics['count_loss']:.6f}, "
        f"mae={final_val_metrics['count_mae']:.4f}, "
        f"acc={final_val_metrics['accuracy']:.4f}",
        flush=True,
    )
    print(
        f"Test  -> loss={final_test_metrics['loss']:.6f}, "
        f"ce={final_test_metrics['ce_loss']:.6f}, "
        f"count={final_test_metrics['count_loss']:.6f}, "
        f"mae={final_test_metrics['count_mae']:.4f}, "
        f"acc={final_test_metrics['accuracy']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()