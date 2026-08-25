from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class ParameterBounds:
    lat_min_deg: float = -89.5
    lat_max_deg: float = 89.5

    lon_min_deg: float = -180.0
    lon_max_deg: float = 180.0

    radius_min_deg: float = 4.0
    radius_max_deg: float = 18.0

    contrast_min: float = 0.3
    contrast_max: float = 0.8

    inclination_min_deg: float = 10.0
    inclination_max_deg: float = 90.0


@dataclass(frozen=True)
class FitConfig:
    n_spots: int
    datapoints: int

    fit_inclination: bool = False
    fixed_inclination_deg: float | None = None

    bounds: ParameterBounds = ParameterBounds()

    # Raw parameter scale before squashing to physical bounds.
    raw_init_scale: float = 2.0
    raw_mutation_scale: float = 0.25

    # Optional penalty to discourage overlapping spots.
    use_separation_penalty: bool = True
    min_spot_gap_deg: float = 2.0
    separation_penalty_weight: float = 1.0

    # JAX / optimizer sizes
    n_candidates: int = 4096
    n_elites: int = 256
    n_steps: int = 250
    seed: int = 0

    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class StrategyResult:
    best_raw_params: object
    best_loss: float
    best_flux: object
    elite_raw_params: object
    elite_losses: object
    loss_history: object


def config_to_jsonable(config: FitConfig) -> dict:
    d = asdict(config)
    d["output_dir"] = str(config.output_dir)
    return d