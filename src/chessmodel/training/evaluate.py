"""Evaluation harness: benchmark a trained checkpoint against its held-out
validation split.

train.py's own validation loop only tracks top-1 policy accuracy, as a
training-time side effect. The Phase 02 exit criteria calls for a documented
benchmark -- top-1 *and* top-3 move accuracy, plus value calibration (Brier
score) -- computed standalone against held-out positions, which is what this
module produces.

Reuses train.py's own train/val split logic (_effective_total_games,
_build_datasets) rather than reimplementing it, so "held-out" here means
exactly the same positions that were excluded from training for a given
checkpoint, not a separately-drawn sample that could overlap.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from chessmodel.data.dataset import DEFAULT_VAL_FRACTION, collate_samples, split_train_val
from chessmodel.training.model import ChessNet
from chessmodel.training.train import _build_datasets, _effective_total_games

_MODEL_SHAPE_KEYS = ("trunk_channels", "num_residual_blocks", "value_channels", "value_hidden")


@dataclass
class EvaluationResult:
    num_positions: int
    top1_accuracy: float
    top3_accuracy: float
    brier_score: float


def _topk_hits(policy_logits: torch.Tensor, policy_targets: torch.Tensor, k: int) -> int:
    """Raw top-k over all policy outputs, unmasked by legal moves -- matches
    train.py's own validation accuracy definition, not select_model_move's
    legal-move-masked one (a different, easier metric)."""
    topk_indices = policy_logits.topk(k, dim=1).indices
    return int((topk_indices == policy_targets.unsqueeze(1)).any(dim=1).sum().item())


def _brier_terms(value_preds: torch.Tensor, value_targets: torch.Tensor) -> torch.Tensor:
    """Brier score is defined on [0, 1] probabilities; value/outcome are on
    [-1, 1] (ADR-0002), so both get remapped before squaring the difference."""
    pred_prob = (value_preds + 1.0) / 2.0
    actual_prob = (value_targets + 1.0) / 2.0
    return (pred_prob - actual_prob) ** 2


def evaluate_checkpoint(
    checkpoint_path: Path,
    dataset_path: Path | None = None,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> EvaluationResult:
    """Evaluate a checkpoint against its own held-out validation split.

    dataset_path overrides the path stored in the checkpoint's config --
    needed whenever the checkpoint was trained somewhere the dataset lived at
    a different path than it does now (e.g. a Kaggle-trained checkpoint
    evaluated locally, where the stored path is /kaggle/input/...).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    stored_config = checkpoint["config"]

    model_shape = {key: stored_config[key] for key in _MODEL_SHAPE_KEYS}
    model = ChessNet(**model_shape).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    resolved_dataset_path = dataset_path or Path(stored_config["dataset_path"])
    max_games = stored_config.get("max_games")
    val_fraction = stored_config.get("val_fraction", DEFAULT_VAL_FRACTION)
    seed = stored_config.get("seed", 0)

    total_games = _effective_total_games(resolved_dataset_path, max_games)
    train_end, _total = split_train_val(total_games, val_fraction)
    _train_dataset, val_dataset = _build_datasets(
        resolved_dataset_path, train_end, total_games, shuffle_buffer_size=0, seed=seed
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_samples)

    total = 0
    top1_hits = 0
    top3_hits = 0
    brier_sum = 0.0

    with torch.no_grad():
        for boards, policy_targets, value_targets in val_loader:
            boards = boards.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)

            policy_logits, value = model(boards)

            top1_hits += _topk_hits(policy_logits, policy_targets, k=1)
            top3_hits += _topk_hits(policy_logits, policy_targets, k=3)
            brier_sum += _brier_terms(value, value_targets).sum().item()
            total += boards.shape[0]

    if total == 0:
        return EvaluationResult(0, 0.0, 0.0, 0.0)
    return EvaluationResult(
        num_positions=total,
        top1_accuracy=top1_hits / total,
        top3_accuracy=top3_hits / total,
        brier_score=brier_sum / total,
    )


def render_benchmark_card(checkpoint_path: Path, result: EvaluationResult) -> str:
    generated_at = datetime.now(UTC).strftime("%Y-%m-%d")
    return (
        f"# Model benchmark: {checkpoint_path.name}\n\n"
        f"Generated {generated_at}. Held-out positions are the chronological "
        f"validation split excluded from this checkpoint's own training run.\n\n"
        f"| Metric | Value |\n"
        f"|---|---|\n"
        f"| Held-out positions evaluated | {result.num_positions:,} |\n"
        f"| Top-1 move accuracy | {result.top1_accuracy:.2%} |\n"
        f"| Top-3 move accuracy | {result.top3_accuracy:.2%} |\n"
        f"| Value calibration (Brier score) | {result.brier_score:.4f} |\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Overrides the dataset path stored in the checkpoint's config",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the benchmark card (default: docs/models/<checkpoint-stem>.md)",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    result = evaluate_checkpoint(args.checkpoint, args.dataset_path, device, args.batch_size)

    output_path = args.output or Path("docs/models") / f"{args.checkpoint.stem}.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_benchmark_card(args.checkpoint, result), encoding="utf-8")

    print(f"Positions evaluated: {result.num_positions}")
    print(f"Top-1 accuracy: {result.top1_accuracy:.4f}")
    print(f"Top-3 accuracy: {result.top3_accuracy:.4f}")
    print(f"Brier score: {result.brier_score:.4f}")
    print(f"Benchmark card written to {output_path}")


if __name__ == "__main__":
    main()
