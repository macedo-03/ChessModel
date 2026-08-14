"""FastAPI serving layer: a containerized inference API for the trained model."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import chess
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chessmodel.serving.play import _MODEL_SHAPE_KEYS, load_model_for_inference, select_model_move


class MoveRequest(BaseModel):
    fen: str


class MoveResponse(BaseModel):
    move_uci: str
    move_san: str
    eval: float
    explanation: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    checkpoint_path: str
    device: str
    trunk_channels: int
    num_residual_blocks: int
    value_channels: int
    value_hidden: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpoint_path_str = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_path_str:
        raise RuntimeError("CHECKPOINT_PATH environment variable must be set")
    checkpoint_path = Path(checkpoint_path_str)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_for_inference(checkpoint_path, device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    shape = {key: checkpoint["config"][key] for key in _MODEL_SHAPE_KEYS}

    app.state.model = model
    app.state.device = device
    app.state.checkpoint_path = checkpoint_path
    app.state.shape = shape
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        checkpoint_path=str(app.state.checkpoint_path),
        device=app.state.device.type,
        **app.state.shape,
    )


@app.post("/move", response_model=MoveResponse)
def move(request: MoveRequest) -> MoveResponse:
    try:
        board = chess.Board(request.fen)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid FEN: {exc}") from exc

    if board.is_game_over():
        raise HTTPException(status_code=422, detail="Position has no legal moves (game over)")

    selected_move, value = select_model_move(app.state.model, board, app.state.device)
    white_perspective_value = value if board.turn == chess.WHITE else -value

    return MoveResponse(
        move_uci=selected_move.uci(),
        move_san=board.san(selected_move),
        eval=white_perspective_value,
        explanation=None,
    )
