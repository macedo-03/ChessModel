"""Generate a tiny, untrained checkpoint for CI's container-integration test.

Not a real trained model -- just enough of a valid checkpoint file (correct
shape, correct config format) to prove the serving container boots, loads a
model, and serves schema-correct responses. Real model quality is verified
elsewhere (the evaluation harness, manual smoke tests against a real trained
checkpoint) -- this script exists purely to keep the CI job fast and free of
external dependencies: no DVC pull, no GPU, no real training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chessmodel.training.model import ChessNet
from chessmodel.training.train import TrainingConfig, save_checkpoint

TINY_MODEL_KWARGS = {
    "trunk_channels": 8,
    "num_residual_blocks": 1,
    "value_channels": 4,
    "value_hidden": 16,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model = ChessNet(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    config = TrainingConfig(
        dataset_path=Path("unused.pgn"),
        checkpoint_dir=args.output.parent,
        mlflow_tracking_uri=str(args.output.parent / "mlruns"),
        trunk_channels=TINY_MODEL_KWARGS["trunk_channels"],
        num_residual_blocks=TINY_MODEL_KWARGS["num_residual_blocks"],
        value_channels=TINY_MODEL_KWARGS["value_channels"],
        value_hidden=TINY_MODEL_KWARGS["value_hidden"],
    )

    save_checkpoint(
        args.output,
        model,
        optimizer,
        scheduler,
        epoch=0,
        global_step=0,
        best_val_loss=1.0,
        config=config,
    )
    print(f"Wrote dummy checkpoint to {args.output}")


if __name__ == "__main__":
    main()
