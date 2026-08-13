"""Local dev/training runner: thin CLI wrapper around train.train().

Kaggle (or any other environment) gets its own separate wrapper supplying its
own paths and secrets -- this one just resolves local defaults and exposes
the TrainingConfig fields that matter for a local run as CLI flags. No
training logic lives here; it's all in train.py's environment-agnostic core.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from chessmodel.training.train import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--mlflow-tracking-uri", type=str, default="mlruns")
    parser.add_argument("--experiment-name", type=str, default="chessmodel")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--trunk-channels", type=int, default=128)
    parser.add_argument("--num-residual-blocks", type=int, default=8)
    args = parser.parse_args()

    config = TrainingConfig(
        dataset_path=args.dataset_path,
        checkpoint_dir=args.checkpoint_dir,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
        max_games=args.max_games,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_workers=args.num_workers,
        device=args.device,
        resume_from=args.resume_from,
        trunk_channels=args.trunk_channels,
        num_residual_blocks=args.num_residual_blocks,
    )

    best_path = train(config)
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
