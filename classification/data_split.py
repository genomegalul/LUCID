from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class ExampleRecord:
    """
    Metadata for one dataset example.
    """
    index: int
    split: SplitName
    label: int          # class index in [0, 4]
    n_spots: int        # original label in [1, 5]
    file_index: int     # global index in the single dataset file


def _resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    """
    Resolve the default dataset location relative to classification/data_split.py.

    Expected layout:
        project_root/
            synthesis/
            plotting/
            classification/
            data/
                dataset_128.npz or dataset_1024.npz
    """
    if data_dir is not None:
        resolved = Path(data_dir).expanduser().resolve()
    else:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        resolved = (project_root / "data").resolve()

    if not resolved.exists():
        raise FileNotFoundError(f"Could not find data directory: {resolved}")

    return resolved


def _resolve_dataset_file(
    data_dir: str | Path | None = None,
    datapoints: int | None = None,
    dataset_file: str | Path | None = None,
) -> Path:
    """
    Resolve the single dataset .npz file.

    If dataset_file is provided, it is used directly.
    Otherwise, datapoints must be provided and the path is:
        data_dir / f"dataset_{datapoints}.npz"
    """
    if dataset_file is not None:
        path = Path(dataset_file).expanduser().resolve()
    else:
        if datapoints is None:
            raise ValueError("Provide either dataset_file or datapoints.")
        root = _resolve_data_dir(data_dir)
        path = root / f"dataset_{datapoints}.npz"

    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {path}")

    return path


def _compute_split_bounds(n_files: int) -> tuple[int, int]:
    """
    Return the exclusive end indices for train and val.

    With the user's guarantee that each class count is divisible by 10,
    this produces exact 80/10/10 splits.
    """
    if n_files <= 0:
        raise ValueError("Cannot split an empty class block.")

    if n_files % 10 != 0:
        raise ValueError(
            f"Expected class count divisible by 10 for exact 80/10/10 split, got {n_files}."
        )

    train_end = (8 * n_files) // 10
    val_end = (9 * n_files) // 10
    return train_end, val_end


def build_split_records(
    data_dir: str | Path | None = None,
    datapoints: int | None = None,
    dataset_file: str | Path | None = None,
) -> dict[SplitName, list[ExampleRecord]]:
    """
    Build deterministic train/val/test records from the single dataset file.

    The file is assumed to be ordered by class blocks:
        all 1-spot examples,
        then all 2-spot examples,
        ...
        then all 5-spot examples.

    Split rule within each class block:
        - first 80%  -> train
        - next 10%   -> val
        - last 10%   -> test
    """
    path = _resolve_dataset_file(
        data_dir=data_dir,
        datapoints=datapoints,
        dataset_file=dataset_file,
    )

    with np.load(path, allow_pickle=False) as data:
        spot_counts = data["spot_counts"].astype(np.int32)
        n_per_category = int(data["n_per_category"])
        labels = data["labels"].astype(np.int64)
        n_spots_all = data["n_spots"].astype(np.int32)

    train_end_offset, val_end_offset = _compute_split_bounds(n_per_category)

    split_records: dict[SplitName, list[ExampleRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for category_index, n_spots in enumerate(spot_counts):
        class_start = category_index * n_per_category

        for offset in range(n_per_category):
            global_index = class_start + offset

            if offset < train_end_offset:
                split: SplitName = "train"
            elif offset < val_end_offset:
                split = "val"
            else:
                split = "test"

            label = int(labels[global_index])
            n_spots_value = int(n_spots_all[global_index])

            record = ExampleRecord(
                index=global_index,
                split=split,
                label=label,
                n_spots=n_spots_value,
                file_index=global_index,
            )

            split_records[split].append(record)

    return split_records


class SpotLightCurveDataset(Dataset):
    """
    PyTorch dataset for spot-count classification from a single large .npz file.

    Input:
        flux or flux_noisy as a float32 tensor of shape (DATAPOINTS,)

    Target:
        class index in [0, 4], corresponding to n_spots in [1, 5]

    Returned item:
        {
            "x": Tensor[DATAPOINTS],
            "y": Tensor[],
            "index": int,
            "n_spots": int,
            "inclination": float,
            "contrast": float,
            "noise": float,
            "theta_deg": Tensor[DATAPOINTS],
        }
    """

    def __init__(
        self,
        split: SplitName,
        data_dir: str | Path | None = None,
        datapoints: int | None = 256,
        dataset_file: str | Path | None = None,
        normalize: bool = True,
        flux_key: str = "flux",
    ) -> None:
        super().__init__()

        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split: {split}")

        if flux_key not in ("flux", "flux_noisy"):
            raise ValueError(f"flux_key must be 'flux' or 'flux_noisy', got {flux_key}")

        self.split = split
        self.normalize = normalize
        self.flux_key = flux_key

        self.dataset_file = _resolve_dataset_file(
            data_dir=data_dir,
            datapoints=datapoints,
            dataset_file=dataset_file,
        )

        self.records = build_split_records(
            data_dir=data_dir,
            datapoints=datapoints,
            dataset_file=self.dataset_file,
        )[split]

        if not self.records:
            raise ValueError(f"No records found for split '{split}' in {self.dataset_file}")

        self.data = np.load(self.dataset_file, allow_pickle=False)

        self.flux = self.data[self.flux_key]
        self.labels = self.data["labels"]
        self.n_spots = self.data["n_spots"]
        self.inclination = self.data["inclination"]
        self.contrast = self.data["contrast"]
        self.noise = self.data["noise"]
        self.theta_deg = self.data["theta_deg"]

        self.datapoints = int(self.data["datapoints"])

    def __len__(self) -> int:
        return len(self.records)

    def _normalize_flux(self, flux: np.ndarray) -> np.ndarray:
        """
        Normalize one light curve using mean centering.
        """
        mean = flux.mean(dtype=np.float32)
        return (flux - mean).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        i = record.index

        flux = self.flux[i].astype(np.float32)

        if self.normalize:
            flux = self._normalize_flux(flux)

        item = {
            "x": torch.from_numpy(flux),
            "y": torch.tensor(int(self.labels[i]), dtype=torch.long),
            "index": int(i),
            "n_spots": int(self.n_spots[i]),
            "inclination": float(self.inclination[i]),
            "contrast": float(self.contrast[i]),
            "noise": float(self.noise[i]),
            "theta_deg": torch.from_numpy(self.theta_deg.astype(np.float32)),
        }

        return item

    def close(self) -> None:
        self.data.close()


def make_datasets(
    data_dir: str | Path | None = None,
    datapoints: int | None = 256,
    dataset_file: str | Path | None = None,
    normalize: bool = True,
    flux_key: str = "flux",
) -> tuple[SpotLightCurveDataset, SpotLightCurveDataset, SpotLightCurveDataset]:
    """
    Convenience constructor for train/val/test datasets.
    """
    train_dataset = SpotLightCurveDataset(
        split="train",
        data_dir=data_dir,
        datapoints=datapoints,
        dataset_file=dataset_file,
        normalize=normalize,
        flux_key=flux_key,
    )
    val_dataset = SpotLightCurveDataset(
        split="val",
        data_dir=data_dir,
        datapoints=datapoints,
        dataset_file=dataset_file,
        normalize=normalize,
        flux_key=flux_key,
    )
    test_dataset = SpotLightCurveDataset(
        split="test",
        data_dir=data_dir,
        datapoints=datapoints,
        dataset_file=dataset_file,
        normalize=normalize,
        flux_key=flux_key,
    )
    return train_dataset, val_dataset, test_dataset


def summarize_splits(
    data_dir: str | Path | None = None,
    datapoints: int | None = 256,
    dataset_file: str | Path | None = None,
) -> dict:
    """
    Return a summary of split sizes for quick inspection/logging.
    """
    path = _resolve_dataset_file(
        data_dir=data_dir,
        datapoints=datapoints,
        dataset_file=dataset_file,
    )

    with np.load(path, allow_pickle=False) as data:
        spot_counts = tuple(int(x) for x in data["spot_counts"])

    records = build_split_records(
        data_dir=data_dir,
        datapoints=datapoints,
        dataset_file=path,
    )

    summary = {
        "dataset_file": str(path),
        "train": {"total": len(records["train"]), "per_class": {}},
        "val": {"total": len(records["val"]), "per_class": {}},
        "test": {"total": len(records["test"]), "per_class": {}},
    }

    for split_name in ("train", "val", "test"):
        per_class = {n_spots: 0 for n_spots in spot_counts}
        for record in records[split_name]:
            per_class[record.n_spots] += 1
        summary[split_name]["per_class"] = per_class

    return summary