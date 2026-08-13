"""Kaggle kernel runner: thin CLI wrapper around train.train().

Kaggle-specific plumbing lives here, not in train.py's environment-agnostic
core:

- The dataset isn't reachable via DVC/R2 from inside a kernel (no
  credentials wired in there) -- it's expected to come from a Kaggle Dataset
  attached to the kernel, mounted under /kaggle/input/<dataset-slug>/. That
  Dataset has to be created and uploaded manually (or via `kaggle datasets
  create`) -- there's nothing automatic about it yet.
- Checkpoints and the MLflow store are written under /kaggle/working/, so
  both get captured as kernel outputs and can be pulled back down afterward
  via `kaggle kernels output <kernel> -p ./pulled`.
- GPU needs no special handling here at all: train()'s device
  auto-detection already picks up CUDA when the kernel has GPU enabled (see
  kernel-metadata.json's "enable_gpu") -- Kaggle just needs to be told to
  turn the accelerator on, the training code doesn't change.

One real limitation this doesn't solve: a kernel's filesystem doesn't
persist between separate runs, so the MLflow sqlite db written under
/kaggle/working/ only covers *this* run. There's no remote tracking server
yet (see the earlier "where does MLflow live" discussion) -- pulling each
run's db down and treating it as a separate local run to inspect is the
current workaround, not a real fix.

This script assumes `chessmodel` isn't already installed in the kernel's
environment and installs it from GitHub on first import -- which requires
the repo to be public (or reachable via a Kaggle-configured git credential)
and the kernel to have internet access enabled.
"""

from __future__ import annotations

try:
    import chessmodel  # noqa: F401
except ImportError:
    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "git+https://github.com/macedo-03/ChessModel.git",
        ]
    )

import argparse
from pathlib import Path

from chessmodel.training.train import TrainingConfig, train

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def resolve_kaggle_dataset_path(dataset_slug: str, dataset_filename: str) -> Path:
    return KAGGLE_INPUT / dataset_slug / dataset_filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-slug",
        default="chessmodel-dataset",
        help="Kaggle Dataset attached to this kernel, mounted under /kaggle/input/<slug>/",
    )
    parser.add_argument(
        "--dataset-filename",
        default="lichess_2026-07_elo2200.pgn",
        help="Filtered PGN filename inside the attached dataset",
    )
    parser.add_argument("--experiment-name", type=str, default="chessmodel")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--trunk-channels", type=int, default=128)
    parser.add_argument("--num-residual-blocks", type=int, default=8)
    args = parser.parse_args()

    config = TrainingConfig(
        dataset_path=resolve_kaggle_dataset_path(args.dataset_slug, args.dataset_filename),
        checkpoint_dir=KAGGLE_WORKING / "checkpoints",
        mlflow_tracking_uri=str(KAGGLE_WORKING / "mlruns"),
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        max_games=args.max_games,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        device=None,  # auto-detect: picks up CUDA automatically when GPU is enabled
        resume_from=args.resume_from,
        trunk_channels=args.trunk_channels,
        num_residual_blocks=args.num_residual_blocks,
    )

    best_path = train(config)
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
