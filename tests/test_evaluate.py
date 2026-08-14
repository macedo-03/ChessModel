"""Tests for the evaluation harness: metric math in isolation, plus one
end-to-end run against a real (tiny) trained checkpoint.
"""

import random
from pathlib import Path

import chess
import torch

from chessmodel.training.evaluate import (
    EvaluationResult,
    _brier_terms,
    _topk_hits,
    evaluate_checkpoint,
    render_benchmark_card,
)
from chessmodel.training.train import TrainingConfig, train

TINY_MODEL_KWARGS = {
    "trunk_channels": 8,
    "num_residual_blocks": 1,
    "value_channels": 4,
    "value_hidden": 16,
}


def _generate_synthetic_pgn(num_games: int, seed: int = 0, max_plies: int = 16) -> str:
    rng = random.Random(seed)
    blocks = []
    for i in range(num_games):
        board = chess.Board()
        sans = []
        for _ in range(rng.randint(4, max_plies)):
            if board.is_game_over():
                break
            move = rng.choice(list(board.legal_moves))
            sans.append(board.san(move))
            board.push(move)
        result = board.result() if board.is_game_over() else rng.choice(["1-0", "0-1", "1/2-1/2"])

        movetext = " ".join(
            f"{idx // 2 + 1}. {san}" if idx % 2 == 0 else san for idx, san in enumerate(sans)
        )
        blocks.append(
            f'[Event "Test {i}"]\n[White "A{i}"]\n[Black "B{i}"]\n[Result "{result}"]\n\n'
            f"{movetext} {result}\n\n"
        )
    return "".join(blocks)


def _write_pgn(tmp_path: Path, num_games: int = 20, seed: int = 0) -> Path:
    path = tmp_path / "games.pgn"
    path.write_text(_generate_synthetic_pgn(num_games, seed=seed), encoding="utf-8")
    return path


# --- metric math, no training needed ----------------------------------------


def test_topk_hits_counts_target_within_top_k() -> None:
    # 3 classes; target index 1 ranks 2nd, 3rd, 3rd across the three rows.
    logits = torch.tensor(
        [
            [3.0, 2.0, 1.0],  # target (1) ranks 2nd
            [1.0, 2.0, 3.0],  # target (1) ranks 2nd
            [3.0, 1.0, 2.0],  # target (1) ranks 3rd (last)
        ]
    )
    targets = torch.tensor([1, 1, 1])

    assert _topk_hits(logits, targets, k=1) == 0
    assert _topk_hits(logits, targets, k=2) == 2
    assert _topk_hits(logits, targets, k=3) == 3


def test_brier_terms_zero_for_perfect_prediction() -> None:
    preds = torch.tensor([1.0, -1.0, 0.0])
    targets = torch.tensor([1.0, -1.0, 0.0])

    terms = _brier_terms(preds, targets)

    assert torch.allclose(terms, torch.zeros_like(terms))


def test_brier_terms_is_max_for_opposite_prediction() -> None:
    preds = torch.tensor([1.0])
    targets = torch.tensor([-1.0])

    terms = _brier_terms(preds, targets)

    assert torch.allclose(terms, torch.tensor([1.0]))


# --- benchmark card rendering, no training needed ---------------------------


def test_render_benchmark_card_includes_all_metrics() -> None:
    result = EvaluationResult(
        num_positions=1000, top1_accuracy=0.512, top3_accuracy=0.78, brier_score=0.1834
    )

    card = render_benchmark_card(Path("best.pt"), result)

    assert "best.pt" in card
    assert "1,000" in card
    assert "51.20%" in card
    assert "78.00%" in card
    assert "0.1834" in card


# --- end-to-end evaluation ---------------------------------------------------


def test_evaluate_checkpoint_produces_sane_metrics(tmp_path: Path) -> None:
    dataset_path = _write_pgn(tmp_path, num_games=20)
    config = TrainingConfig(
        dataset_path=dataset_path,
        checkpoint_dir=tmp_path / "checkpoints",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        batch_size=8,
        num_epochs=1,
        warmup_steps=2,
        num_workers=0,
        device="cpu",
        **TINY_MODEL_KWARGS,
    )
    best_path = train(config)

    result = evaluate_checkpoint(best_path, dataset_path=dataset_path, device=torch.device("cpu"))

    assert result.num_positions > 0
    assert 0.0 <= result.top1_accuracy <= result.top3_accuracy <= 1.0
    assert 0.0 <= result.brier_score <= 1.0


def test_evaluate_checkpoint_uses_stored_dataset_path_by_default(tmp_path: Path) -> None:
    dataset_path = _write_pgn(tmp_path, num_games=20)
    config = TrainingConfig(
        dataset_path=dataset_path,
        checkpoint_dir=tmp_path / "checkpoints",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        batch_size=8,
        num_epochs=1,
        warmup_steps=2,
        num_workers=0,
        device="cpu",
        **TINY_MODEL_KWARGS,
    )
    best_path = train(config)

    # No dataset_path override -- must fall back to the (now string) path
    # stored in the checkpoint's own config.
    result = evaluate_checkpoint(best_path, device=torch.device("cpu"))

    assert result.num_positions > 0
