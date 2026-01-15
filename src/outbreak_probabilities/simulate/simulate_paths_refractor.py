# src/outbreak-probabilities/simulate/simulate_paths_refractor.py
"""
Refactor of your test script into a reusable function and a small CLI entrypoint.
Assumes compute_serial_weights and generate_batch are importable from the package.
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import numpy as np
import logging
import pathlib

# Import API 
from .calculate_serial_weights import compute_serial_weights
from .batch_processing import generate_batch

# Start logger
logger = logging.getLogger(__name__)

@dataclass
class SimConfig:
    N: int = 1000
    max_weeks: int = 50
    mean_serial: float = 15.3
    std_serial: float = 9.3
    k_max: int = 50
    nquad: int = 32
    step: float = 7.0
    R_range: Tuple[float, float] = (0.0, 10.0)
    initial_cases: Iterable[int] = (1,)
    extinction_window: int = 10
    major_threshold: int = 100
    seed: Optional[int] = None
    R_dist: str = "uniform"
    R_dist_params: Optional[dict] = None
    out_path: str = "data/test_simulations.csv"
    use_tempfile: bool = False
    generate_full: bool = False
    write_weeks: int = 5


def prepare_serial_weights(
    mean: float,
    std: float,
    k_max: int,
    nquad: int,
    step: float,
):
    """Wrap compute_serial_weights for clarity and unit testing."""
    w = compute_serial_weights(mean=mean, std=std, k_max=k_max, nquad=nquad, step=step)
    logger.debug("Computed serial weights (len=%d)", len(w))
    return w


def simulate_batch(cfg: SimConfig):
    """Run the batch generation and return the trajectories and csv path.

    This wraps your generate_batch call so calling code can import and run it.
    """
    # Convert out_path to ensure directory exists
    out_path = pathlib.Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    w = prepare_serial_weights(
        mean=cfg.mean_serial,
        std=cfg.std_serial,
        k_max=cfg.k_max,
        nquad=cfg.nquad,
        step=cfg.step,
    )

    trajectories, csv_path = generate_batch(
        N=int(cfg.N),
        w=w,
        max_weeks=cfg.max_weeks,
        R_range=cfg.R_range,
        initial_cases=list(cfg.initial_cases),
        extinction_window=cfg.extinction_window,
        major_threshold=cfg.major_threshold,
        seed=cfg.seed,
        R_dist=cfg.R_dist,
        R_dist_params=cfg.R_dist_params,
        out_path=str(out_path),
        use_tempfile=cfg.use_tempfile,
        generate_full=cfg.generate_full,
        write_weeks=cfg.write_weeks,
    )

    logger.info("Simulated trajectories shape: %s", getattr(trajectories, "shape", repr(trajectories)))
    logger.info("CSV written to: %s", csv_path)
    return trajectories, csv_path


def main(argv=None):
    """Use this part to check file"""
    import argparse

    parser = argparse.ArgumentParser(description="Run a test batch of simulated trajectories.")
    parser.add_argument("--N", type=int, default=1000, help="Number of trajectories to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Master/random seed")
    parser.add_argument("--out", type=str, default="data/test_simulations.csv", help="Output CSV path")
    args = parser.parse_args(argv)

    cfg = SimConfig(N=args.N, seed=args.seed, out_path=args.out, use_tempfile=args.use_tempfile)
    simulate_batch(cfg)
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
