from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_split import SpotLightCurveDataset
from model import SpotCountCNN


# --- Configuration ---

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
PLOTS_DIR = SCRIPT_DIR / "plots"
WEIGHTS_DIR = SCRIPT_DIR / "weights"
DATA_DIR = SCRIPT_DIR.parent / "data"

WEIGHTS_PATH = WEIGHTS_DIR / "weights_20260429_195531.pt"

BATCH_SIZE = 512
NUM_WORKERS = 4
NORMALIZE_INPUT = True

# Match the dataset you trained this checkpoint on.
DATAPOINTS = 256
FLUX_KEY = "flux"

# Inclination bins in degrees: [10,20), [20,30), ..., [80,90]
INCLINATION_BIN_EDGES = np.arange(10.0, 100.0, 10.0, dtype=np.float32)


# --- Helpers ---

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(weights_path: Path, device: torch.device) -> tuple[SpotCountCNN, dict]:
    checkpoint = torch.load(weights_path, map_location=device)

    model_kwargs = checkpoint.get(
        "model_kwargs",
        {
            "input_length": DATAPOINTS,
            "n_classes": 5,
        },
    )

    model = SpotCountCNN(**model_kwargs).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint


def move_batch_to_device(
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device=device, dtype=torch.float32)
    inc = batch["inclination"].to(device=device, dtype=torch.float32)
    y = batch["y"].to(device=device, dtype=torch.long)
    return x, inc, y


def forward_model(
    model: SpotCountCNN,
    x: torch.Tensor,
    inc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Supports both:
      - latest multitask model with forward_with_count()
      - older categorical-only model with forward()
    """
    if hasattr(model, "forward_with_count"):
        logits, count_pred = model.forward_with_count(x, inc)
        return logits, count_pred

    logits = model(x, inc)
    return logits, None


def compute_bin_rows(
    inclinations_deg: np.ndarray,
    correct: np.ndarray,
    bin_edges: np.ndarray,
) -> list[dict]:
    rows = []

    for i in range(len(bin_edges) - 1):
        lo = float(bin_edges[i])
        hi = float(bin_edges[i + 1])

        if i == len(bin_edges) - 2:
            mask = (inclinations_deg >= lo) & (inclinations_deg <= hi)
            label = f"[{lo:.0f}, {hi:.0f}]"
        else:
            mask = (inclinations_deg >= lo) & (inclinations_deg < hi)
            label = f"[{lo:.0f}, {hi:.0f})"

        n = int(mask.sum())
        n_correct = int(correct[mask].sum()) if n > 0 else 0
        acc = float(n_correct / n) if n > 0 else float("nan")

        rows.append(
            {
                "bin_label": label,
                "inclination_lo_deg": lo,
                "inclination_hi_deg": hi,
                "count": n,
                "n_correct": n_correct,
                "accuracy": acc,
                "bin_center_deg": 0.5 * (lo + hi),
            }
        )

    return rows


def compute_count_bin_rows(
    inclinations_deg: np.ndarray,
    count_abs_error: np.ndarray,
    bin_edges: np.ndarray,
) -> list[dict]:
    rows = []

    for i in range(len(bin_edges) - 1):
        lo = float(bin_edges[i])
        hi = float(bin_edges[i + 1])

        if i == len(bin_edges) - 2:
            mask = (inclinations_deg >= lo) & (inclinations_deg <= hi)
            label = f"[{lo:.0f}, {hi:.0f}]"
        else:
            mask = (inclinations_deg >= lo) & (inclinations_deg < hi)
            label = f"[{lo:.0f}, {hi:.0f})"

        n = int(mask.sum())
        mae = float(count_abs_error[mask].mean()) if n > 0 else float("nan")

        rows.append(
            {
                "bin_label": label,
                "inclination_lo_deg": lo,
                "inclination_hi_deg": hi,
                "count": n,
                "count_mae": mae,
                "bin_center_deg": 0.5 * (lo + hi),
            }
        )

    return rows


def compute_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_classes: int = 5,
) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int32)
    for true_label, pred_label in zip(targets, predictions):
        matrix[true_label - 1, pred_label - 1] += 1
    return matrix


def compute_per_class_accuracy(
    targets: np.ndarray,
    predictions: np.ndarray,
    n_classes: int = 5,
) -> dict:
    result = {}

    for class_label in range(1, n_classes + 1):
        mask = targets == class_label
        n = int(mask.sum())
        n_correct = int((predictions[mask] == class_label).sum()) if n > 0 else 0
        acc = float(n_correct / n) if n > 0 else float("nan")

        result[str(class_label)] = {
            "count": n,
            "n_correct": n_correct,
            "accuracy": acc,
        }

    return result


def compute_extreme_error_summary(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict:
    """
    Track the specific edge-attraction errors you were interested in:
      true 2 -> pred 1
      true 3 -> pred 5
      true 4 -> pred 5
    """
    summary = {}

    cases = [
        ("true_2_pred_1", 2, 1),
        ("true_3_pred_5", 3, 5),
        ("true_4_pred_5", 4, 5),
    ]

    for name, true_label, pred_label in cases:
        true_mask = targets == true_label
        n_true = int(true_mask.sum())
        n_error = int(((targets == true_label) & (predictions == pred_label)).sum())
        frac = float(n_error / n_true) if n_true > 0 else float("nan")

        summary[name] = {
            "true_label": true_label,
            "pred_label": pred_label,
            "n_true": n_true,
            "n_error": n_error,
            "fraction_of_true_class": frac,
        }

    return summary


def plot_inclination_accuracy(
    rows: list[dict],
    plot_path: Path,
) -> None:
    x_vals = np.array([row["bin_center_deg"] for row in rows], dtype=np.float32)
    y_vals = np.array([row["accuracy"] for row in rows], dtype=np.float32)
    counts = np.array([row["count"] for row in rows], dtype=np.int32)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel("Inclination (deg, bin center)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy by inclination bin (test set)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)

    for x, y, n in zip(x_vals, y_vals, counts):
        if np.isfinite(y):
            plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_count_mae_by_inclination(
    rows: list[dict],
    plot_path: Path,
) -> None:
    x_vals = np.array([row["bin_center_deg"] for row in rows], dtype=np.float32)
    y_vals = np.array([row["count_mae"] for row in rows], dtype=np.float32)
    counts = np.array([row["count"] for row in rows], dtype=np.int32)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, marker="o")
    plt.xlabel("Inclination (deg, bin center)")
    plt.ylabel("Count MAE")
    plt.title("Count-regression MAE by inclination bin (test set)")
    plt.grid(True, alpha=0.3)

    for x, y, n in zip(x_vals, y_vals, counts):
        if np.isfinite(y):
            plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(
    matrix: np.ndarray,
    plot_path: Path,
) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, interpolation="nearest", aspect="equal")
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted number of spots")
    plt.ylabel("True number of spots")
    plt.xticks(np.arange(5), [1, 2, 3, 4, 5])
    plt.yticks(np.arange(5), [1, 2, 3, 4, 5])
    plt.colorbar()

    max_val = matrix.max() if matrix.size > 0 else 0
    threshold = max_val / 2.0 if max_val > 0 else 0.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            text_color = "white" if val > threshold else "black"
            plt.text(j, i, str(val), ha="center", va="center", color=text_color)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_count_prediction_scatter(
    targets: np.ndarray,
    count_predictions: np.ndarray,
    plot_path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.scatter(targets, count_predictions, s=4, alpha=0.15)
    plt.plot([1, 5], [1, 5], linestyle="--")
    plt.xlabel("True number of spots")
    plt.ylabel("Predicted continuous count")
    plt.title("Count-regression predictions (test set)")
    plt.xlim(0.75, 5.25)
    plt.ylim(0.75, 5.25)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()


# --- Main ---

def main() -> None:
    ensure_dirs()

    device = get_device()
    timestamp = make_timestamp()

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Could not find weights file: {WEIGHTS_PATH}")

    print(f"Using device: {device}", flush=True)
    print(f"Weights: {WEIGHTS_PATH}", flush=True)

    model, checkpoint = load_model(WEIGHTS_PATH, device)

    train_config = checkpoint.get("train_config", {})
    datapoints = int(train_config.get("datapoints", checkpoint.get("model_kwargs", {}).get("input_length", DATAPOINTS)))
    flux_key = str(train_config.get("flux_key", FLUX_KEY))

    print(f"Datapoints: {datapoints}", flush=True)
    print(f"Flux key: {flux_key}", flush=True)

    test_dataset = SpotLightCurveDataset(
        split="test",
        data_dir=DATA_DIR,
        datapoints=datapoints,
        normalize=NORMALIZE_INPUT,
        flux_key=flux_key,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    all_inclinations = []
    all_correct = []
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_count_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            x, inc, y = move_batch_to_device(batch, device)

            logits, count_pred = forward_model(model, x, inc)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            correct = preds == y

            all_inclinations.extend(batch["inclination"].tolist())
            all_correct.extend(correct.cpu().numpy().astype(np.int32).tolist())
            all_predictions.extend((preds.cpu().numpy() + 1).tolist())
            all_targets.extend((y.cpu().numpy() + 1).tolist())
            all_probabilities.extend(probs.cpu().tolist())

            if count_pred is not None:
                all_count_predictions.extend(count_pred.cpu().tolist())

    inclinations_deg = np.asarray(all_inclinations, dtype=np.float32)
    correct = np.asarray(all_correct, dtype=np.int32)
    predictions = np.asarray(all_predictions, dtype=np.int32)
    targets = np.asarray(all_targets, dtype=np.int32)
    probabilities = np.asarray(all_probabilities, dtype=np.float32)

    has_count_predictions = len(all_count_predictions) == len(targets)
    if has_count_predictions:
        count_predictions = np.asarray(all_count_predictions, dtype=np.float32)
        count_abs_error = np.abs(count_predictions - targets.astype(np.float32))
        overall_count_mae = float(count_abs_error.mean())
    else:
        count_predictions = None
        count_abs_error = None
        overall_count_mae = None

    rows = compute_bin_rows(
        inclinations_deg=inclinations_deg,
        correct=correct,
        bin_edges=INCLINATION_BIN_EDGES,
    )

    if has_count_predictions:
        count_rows = compute_count_bin_rows(
            inclinations_deg=inclinations_deg,
            count_abs_error=count_abs_error,
            bin_edges=INCLINATION_BIN_EDGES,
        )
    else:
        count_rows = []

    confusion_matrix = compute_confusion_matrix(
        targets=targets,
        predictions=predictions,
        n_classes=5,
    )

    per_class_accuracy = compute_per_class_accuracy(
        targets=targets,
        predictions=predictions,
        n_classes=5,
    )

    extreme_error_summary = compute_extreme_error_summary(
        targets=targets,
        predictions=predictions,
    )

    overall_acc = float(correct.mean())

    print(f"\nOverall test accuracy: {overall_acc:.4f}", flush=True)
    if has_count_predictions:
        print(f"Overall count MAE:     {overall_count_mae:.4f}", flush=True)
    print("", flush=True)

    print("Accuracy by inclination bin:", flush=True)
    for row in rows:
        acc_str = f"{row['accuracy']:.4f}" if np.isfinite(row["accuracy"]) else "nan"
        print(
            f"  {row['bin_label']:>10}  "
            f"count={row['count']:5d}  "
            f"correct={row['n_correct']:5d}  "
            f"acc={acc_str}",
            flush=True,
        )

    if has_count_predictions:
        print("\nCount MAE by inclination bin:", flush=True)
        for row in count_rows:
            mae_str = f"{row['count_mae']:.4f}" if np.isfinite(row["count_mae"]) else "nan"
            print(
                f"  {row['bin_label']:>10}  "
                f"count={row['count']:5d}  "
                f"mae={mae_str}",
                flush=True,
            )

    print("\nPer-class accuracy:", flush=True)
    for class_label, stats in per_class_accuracy.items():
        print(
            f"  {class_label} spot(s): "
            f"count={stats['count']:5d}  "
            f"correct={stats['n_correct']:5d}  "
            f"acc={stats['accuracy']:.4f}",
            flush=True,
        )

    print("\nSpecific extreme-error summary:", flush=True)
    for name, stats in extreme_error_summary.items():
        print(
            f"  {name}: "
            f"{stats['n_error']}/{stats['n_true']} "
            f"= {stats['fraction_of_true_class']:.4f}",
            flush=True,
        )

    print("\nConfusion matrix (rows=true, cols=pred):", flush=True)
    print(confusion_matrix, flush=True)

    csv_path = RUNS_DIR / f"inclination_metrics_{timestamp}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "bin_label",
                "inclination_lo_deg",
                "inclination_hi_deg",
                "bin_center_deg",
                "count",
                "n_correct",
                "accuracy",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if has_count_predictions:
        count_csv_path = RUNS_DIR / f"count_mae_by_inclination_{timestamp}.csv"
        with count_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "bin_label",
                    "inclination_lo_deg",
                    "inclination_hi_deg",
                    "bin_center_deg",
                    "count",
                    "count_mae",
                ],
            )
            writer.writeheader()
            for row in count_rows:
                writer.writerow(row)
    else:
        count_csv_path = None

    summary_path = RUNS_DIR / f"inclination_metrics_{timestamp}.json"
    summary_payload = {
        "weights_path": str(WEIGHTS_PATH),
        "datapoints": datapoints,
        "flux_key": flux_key,
        "overall_test_accuracy": overall_acc,
        "overall_count_mae": overall_count_mae,
        "n_test_examples": int(len(correct)),
        "inclination_rows": rows,
        "count_mae_by_inclination": count_rows,
        "per_class_accuracy": per_class_accuracy,
        "extreme_error_summary": extreme_error_summary,
        "confusion_matrix": confusion_matrix.tolist(),
        "class_labels": [1, 2, 3, 4, 5],
    }

    if has_count_predictions:
        summary_payload["count_prediction_summary"] = {
            "mean": float(count_predictions.mean()),
            "std": float(count_predictions.std()),
            "min": float(count_predictions.min()),
            "max": float(count_predictions.max()),
        }

    with summary_path.open("w") as f:
        json.dump(summary_payload, f, indent=2)

    outputs_path = RUNS_DIR / f"test_predictions_{timestamp}.json"
    outputs_payload = {
        "weights_path": str(WEIGHTS_PATH),
        "datapoints": datapoints,
        "flux_key": flux_key,
        "targets": targets.tolist(),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "inclinations": inclinations_deg.tolist(),
    }

    if has_count_predictions:
        outputs_payload["count_predictions"] = count_predictions.tolist()

    with outputs_path.open("w") as f:
        json.dump(outputs_payload, f, indent=2)

    inclination_plot_path = PLOTS_DIR / f"inclination_accuracy_{timestamp}.png"
    plot_inclination_accuracy(rows, inclination_plot_path)

    confusion_plot_path = PLOTS_DIR / f"confusion_matrix_{timestamp}.png"
    plot_confusion_matrix(confusion_matrix, confusion_plot_path)

    if has_count_predictions:
        count_mae_plot_path = PLOTS_DIR / f"count_mae_by_inclination_{timestamp}.png"
        plot_count_mae_by_inclination(count_rows, count_mae_plot_path)

        count_scatter_path = PLOTS_DIR / f"count_prediction_scatter_{timestamp}.png"
        plot_count_prediction_scatter(targets, count_predictions, count_scatter_path)
    else:
        count_mae_plot_path = None
        count_scatter_path = None

    print(f"\nSaved CSV:                   {csv_path}", flush=True)
    if count_csv_path is not None:
        print(f"Saved count MAE CSV:         {count_csv_path}", flush=True)
    print(f"Saved JSON:                  {summary_path}", flush=True)
    print(f"Saved predictions JSON:      {outputs_path}", flush=True)
    print(f"Saved inclination plot:      {inclination_plot_path}", flush=True)
    print(f"Saved confusion plot:        {confusion_plot_path}", flush=True)
    if count_mae_plot_path is not None:
        print(f"Saved count MAE plot:        {count_mae_plot_path}", flush=True)
    if count_scatter_path is not None:
        print(f"Saved count scatter plot:    {count_scatter_path}", flush=True)


if __name__ == "__main__":
    main()