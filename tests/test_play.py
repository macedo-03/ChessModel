"""Tests for the interactive play script's pure logic.

Not testing play()/main() themselves -- those are the input()-driven
interactive loop, consistent with how the other thin CLI wrappers in this
project (ingest.py, run_local.py, run_kaggle.py) leave main() untested and
only lock in the actual logic underneath it.
"""

import re
from pathlib import Path

import chess
import torch

from chessmodel.serving.play import _render_board, load_model_for_inference, select_model_move
from chessmodel.training.model import ChessNet
from chessmodel.training.train import TrainingConfig, save_checkpoint

_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


TINY_MODEL_KWARGS = {
    "trunk_channels": 8,
    "num_residual_blocks": 1,
    "value_channels": 4,
    "value_hidden": 16,
}


def _save_tiny_checkpoint(tmp_path: Path) -> Path:
    model = ChessNet(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)
    config = TrainingConfig(
        dataset_path=tmp_path / "unused.pgn",
        checkpoint_dir=tmp_path,
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        **TINY_MODEL_KWARGS,
    )

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, optimizer, scheduler, epoch=0, global_step=0, best_val_loss=1.0, config=config
    )
    return path, model


def test_load_model_for_inference_reconstructs_architecture_and_weights(tmp_path: Path) -> None:
    path, original_model = _save_tiny_checkpoint(tmp_path)

    loaded_model = load_model_for_inference(path, device=torch.device("cpu"))

    assert not loaded_model.training  # eval() was called
    for original, loaded in zip(
        original_model.parameters(), loaded_model.parameters(), strict=True
    ):
        assert torch.equal(original, loaded)


def test_load_model_for_inference_produces_correct_output_shape(tmp_path: Path) -> None:
    path, _model = _save_tiny_checkpoint(tmp_path)
    loaded_model = load_model_for_inference(path, device=torch.device("cpu"))

    batch = torch.randn(1, 19, 8, 8)
    with torch.no_grad():
        policy_logits, value = loaded_model(batch)

    assert policy_logits.shape == (1, 8 * 8 * 73)
    assert value.shape == (1,)


def test_select_model_move_returns_a_legal_move(tmp_path: Path) -> None:
    path, _model = _save_tiny_checkpoint(tmp_path)
    model = load_model_for_inference(path, device=torch.device("cpu"))
    board = chess.Board()

    move, value = select_model_move(model, board, device=torch.device("cpu"))

    assert move in board.legal_moves
    assert -1.0 <= value <= 1.0


def test_select_model_move_is_deterministic(tmp_path: Path) -> None:
    path, _model = _save_tiny_checkpoint(tmp_path)
    model = load_model_for_inference(path, device=torch.device("cpu"))
    board = chess.Board()

    move1, _ = select_model_move(model, board, device=torch.device("cpu"))
    move2, _ = select_model_move(model, board, device=torch.device("cpu"))

    assert move1 == move2


_WHITE_GLYPHS = set("♙♘♗♖♕♔")
_BLACK_GLYPHS = set("♟♞♝♜♛♚")


def test_render_board_uses_distinct_glyphs_for_white_black() -> None:
    board = chess.Board()
    plain = _strip_ansi(_render_board(board))

    assert any(glyph in plain for glyph in _WHITE_GLYPHS)
    assert any(glyph in plain for glyph in _BLACK_GLYPHS)


def test_render_board_cells_are_uniform_width() -> None:
    board = chess.Board()
    plain = _strip_ansi(_render_board(board))

    board_rows = plain.splitlines()[1:-1]  # drop the file-label header/footer rows
    assert len(board_rows) == 16  # 2 lines per rank x 8 ranks, for taller squares
    row_lengths = {len(row) for row in board_rows}
    assert len(row_lengths) == 1  # every line is exactly the same length


def test_render_board_glyph_count_matches_piece_count() -> None:
    # A starting position has exactly 32 pieces -- if any glyph leaked into
    # (or was missing from) the wrong cell, this count would drift.
    board = chess.Board()
    plain = _strip_ansi(_render_board(board))

    all_glyphs = _WHITE_GLYPHS | _BLACK_GLYPHS
    glyph_count = sum(plain.count(glyph) for glyph in all_glyphs)
    assert glyph_count == 32
