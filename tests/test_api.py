"""In-process tests for the serving API: schema, error handling, and the
White-perspective eval normalization -- no Docker involved.

The real "boot the container" integration test is a separate CI concern (see
Phase 04 planning); these tests only need CHECKPOINT_PATH pointed at a tiny
untrained checkpoint, same fixture pattern as test_play.py.
"""

from pathlib import Path

import chess
import pytest
import torch
from fastapi.testclient import TestClient

from chessmodel.training.model import ChessNet
from chessmodel.training.train import TrainingConfig, save_checkpoint

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
    return path


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    checkpoint_path = _save_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint_path))

    from chessmodel.serving.api import app

    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_returns_shape_and_device(client: TestClient) -> None:
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()
    assert body["trunk_channels"] == TINY_MODEL_KWARGS["trunk_channels"]
    assert body["num_residual_blocks"] == TINY_MODEL_KWARGS["num_residual_blocks"]
    assert body["value_channels"] == TINY_MODEL_KWARGS["value_channels"]
    assert body["value_hidden"] == TINY_MODEL_KWARGS["value_hidden"]
    assert body["device"] == "cpu"


def test_move_returns_a_legal_move(client: TestClient) -> None:
    response = client.post("/move", json={"fen": chess.Board().fen()})

    assert response.status_code == 200
    body = response.json()

    board = chess.Board()
    move = chess.Move.from_uci(body["move_uci"])
    assert move in board.legal_moves
    assert board.san(move) == body["move_san"]
    assert -1.0 <= body["eval"] <= 1.0
    assert body["explanation"] is None


def test_move_eval_is_normalized_to_white_perspective(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mock select_model_move to return a known, fixed mover's-perspective
    # value -- isolates the sign-flip logic from actual model numerics (which
    # can differ by a tiny float epsilon between threads under PyTorch's CPU
    # intra-op parallelism, an irrelevant source of noise for this check).
    fixed_value = 0.4

    def fake_select_model_move(
        model: object, board: chess.Board, device: object
    ) -> tuple[chess.Move, float]:
        return next(iter(board.legal_moves)), fixed_value

    monkeypatch.setattr("chessmodel.serving.api.select_model_move", fake_select_model_move)

    white_response = client.post("/move", json={"fen": chess.Board().fen()})
    assert white_response.json()["eval"] == pytest.approx(fixed_value)

    black_board = chess.Board()
    black_board.push_san("e4")
    black_response = client.post("/move", json={"fen": black_board.fen()})
    assert black_response.json()["eval"] == pytest.approx(-fixed_value)


def test_move_rejects_invalid_fen(client: TestClient) -> None:
    response = client.post("/move", json={"fen": "not a real fen"})

    assert response.status_code == 400


def test_move_rejects_game_over_position(client: TestClient) -> None:
    board = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        board.push_san(san)

    response = client.post("/move", json={"fen": board.fen()})

    assert response.status_code == 422
