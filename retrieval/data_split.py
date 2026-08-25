from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class RetrievalRecord:
    index: int
    split: SplitName
    n_spots: int
    label: int


class RetrievalDataStore:
    """
    Shared access to the single large dataset file.

    Expected file:
        ../data/dataset_{datapoints}.npz

    Expected arrays:
        flux            (N, datapoints)
        flux_noisy      (N, datapoints)
        theta_deg       (datapoints,)
        n_spots         (N,)
        labels          (N,)
        noise           (N,)
        inclination     (N,)
        contrast        (N,)
        lats/lons/radii (N, max_spots)
    """

    def __init__(
        self,
        dataset_file: str | Path,
        flux_key: str = "flux",
    ) -> None:
        self.dataset_file = Path(dataset_file).expanduser().resolve()

        if not self.dataset_file.exists():
            raise FileNotFoundError(f"Could not find dataset file: {self.dataset_file}")

        if flux_key not in ("flux", "flux_noisy"):
            raise ValueError(f"flux_key must be 'flux' or 'flux_noisy', got {flux_key}")

        self.flux_key = flux_key
        self.data = np.load(self.dataset_file, allow_pickle=False)

        self.flux = self.data[flux_key]
        self.theta_deg = self.data["theta_deg"]

        self.n_spots = self.data["n_spots"]
        self.labels = self.data["labels"]
        self.noise = self.data["noise"]
        self.inclination = self.data["inclination"]
        self.contrast = self.data["contrast"]

        self.lats = self.data["lats"]
        self.lons = self.data["lons"]
        self.radii = self.data["radii"]

        self.datapoints = int(self.data["datapoints"])
        self.n_datasets = int(self.data["n_datasets"])
        self.n_per_category = int(self.data["n_per_category"])
        self.spot_counts = tuple(int(x) for x in self.data["spot_counts"])
        self.max_spots = int(self.data["max_spots"])

    def close(self) -> None:
        self.data.close()


def resolve_dataset_file(
    data_dir: str | Path | None = None,
    datapoints: int = 256,
    dataset_file: str | Path | None = None,
) -> Path:
    if dataset_file is not None:
        path = Path(dataset_file).expanduser().resolve()
    else:
        if data_dir is None:
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            data_dir = project_root / "data"

        path = Path(data_dir).expanduser().resolve() / f"dataset_{datapoints}.npz"

    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {path}")

    return path


def compute_split_offsets(n_per_category: int) -> tuple[int, int]:
    if n_per_category <= 0:
        raise ValueError("n_per_category must be positive.")

    if n_per_category % 10 != 0:
        raise ValueError(
            f"n_per_category must be divisible by 10 for exact 80/10/10 split, got {n_per_category}."
        )

    train_end = (8 * n_per_category) // 10
    val_end = (9 * n_per_category) // 10
    return train_end, val_end


def build_records_for_spot_count(
    store: RetrievalDataStore,
    n_spots: int,
) -> dict[SplitName, list[RetrievalRecord]]:
    if n_spots not in store.spot_counts:
        raise ValueError(f"n_spots={n_spots} not in dataset spot_counts={store.spot_counts}")

    category_index = list(store.spot_counts).index(n_spots)
    class_start = category_index * store.n_per_category

    train_end, val_end = compute_split_offsets(store.n_per_category)

    records: dict[SplitName, list[RetrievalRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }

    for offset in range(store.n_per_category):
        global_index = class_start + offset

        if offset < train_end:
            split: SplitName = "train"
        elif offset < val_end:
            split = "val"
        else:
            split = "test"

        records[split].append(
            RetrievalRecord(
                index=global_index,
                split=split,
                n_spots=int(store.n_spots[global_index]),
                label=int(store.labels[global_index]),
            )
        )

    return records


class RetrievalDataset(Dataset):
    """
    Dataset for one retrieval model with fixed n_spots.

    Returns:
        (lc_in, aux), target_lc, metadata

    where:
        lc_in      shape: (1, datapoints), normalized input curve
        aux        shape: (2,), [sin(inclination), cos(inclination)]
        target_lc  shape: (datapoints,), target clean/noisy flux
    """

    def __init__(
        self,
        store: RetrievalDataStore,
        records: list[RetrievalRecord],
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.store = store
        self.records = records
        self.normalize_input = normalize_input

        if len(self.records) == 0:
            raise ValueError("Empty RetrievalDataset.")

    def __len__(self) -> int:
        return len(self.records)

    def _normalize_curve(self, flux: np.ndarray) -> np.ndarray:
        mean = flux.mean(dtype=np.float32)
        std = flux.std(dtype=np.float32)
        std = max(float(std), 1e-7)
        return ((flux - mean) / std).astype(np.float32)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        i = record.index

        target_lc = np.nan_to_num(self.store.flux[i].astype(np.float32), nan=1.0)

        if self.normalize_input:
            lc_in = self._normalize_curve(target_lc)
        else:
            lc_in = target_lc.astype(np.float32)

        inc_deg = float(self.store.inclination[i])
        inc_rad = np.deg2rad(inc_deg)

        aux = np.array(
            [np.sin(inc_rad), np.cos(inc_rad)],
            dtype=np.float32,
        )

        metadata = {
            "index": int(i),
            "n_spots": int(self.store.n_spots[i]),
            "inclination": inc_deg,
            "contrast": float(self.store.contrast[i]),
            "noise": float(self.store.noise[i]),
        }

        return (
            torch.from_numpy(lc_in).unsqueeze(0),
            torch.from_numpy(aux),
        ), torch.from_numpy(target_lc), metadata


def make_datasets(
    n_spots: int,
    data_dir: str | Path | None = None,
    datapoints: int = 256,
    dataset_file: str | Path | None = None,
    flux_key: str = "flux",
    normalize_input: bool = True,
) -> tuple[RetrievalDataset, RetrievalDataset, RetrievalDataset, RetrievalDataStore]:
    path = resolve_dataset_file(
        data_dir=data_dir,
        datapoints=datapoints,
        dataset_file=dataset_file,
    )

    store = RetrievalDataStore(
        dataset_file=path,
        flux_key=flux_key,
    )

    records = build_records_for_spot_count(
        store=store,
        n_spots=n_spots,
    )

    train_dataset = RetrievalDataset(
        store=store,
        records=records["train"],
        normalize_input=normalize_input,
    )
    val_dataset = RetrievalDataset(
        store=store,
        records=records["val"],
        normalize_input=normalize_input,
    )
    test_dataset = RetrievalDataset(
        store=store,
        records=records["test"],
        normalize_input=normalize_input,
    )

    return train_dataset, val_dataset, test_dataset, store


def summarize_split(
    n_spots: int,
    data_dir: str | Path | None = None,
    datapoints: int = 256,
    dataset_file: str | Path | None = None,
    flux_key: str = "flux",
) -> dict:
    path = resolve_dataset_file(
        data_dir=data_dir,
        datapoints=datapoints,
        dataset_file=dataset_file,
    )

    store = RetrievalDataStore(path, flux_key=flux_key)
    records = build_records_for_spot_count(store, n_spots=n_spots)

    summary = {
        "dataset_file": str(path),
        "datapoints": store.datapoints,
        "flux_key": flux_key,
        "n_spots": n_spots,
        "train": len(records["train"]),
        "val": len(records["val"]),
        "test": len(records["test"]),
    }

    store.close()
    return summary