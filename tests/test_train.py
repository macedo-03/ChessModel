"""Tests for the training loop: an end-to-end run against a synthetic PGN,
checkpoint round-tripping, resume, and the LR schedule's math in isolation.

MLflow points at a local SQLite database under tmp_path throughout -- no
server, no network, matching how the loop is expected to run in CI or a
quick local sanity check.
"""

import random
from pathlib import Path

import chess
import pytest
import torch
from mlflow.tracking import MlflowClient

from chessmodel.training.model import ChessNet
from chessmodel.training.train import (
    REGISTERED_MODEL_NAME,
    TrainingConfig,
    _build_datasets,
    _effective_total_games,
    _estimate_total_steps,
    _resolve_mlflow_tracking_uri,
    _warmup_cosine_lr_lambda,
    load_checkpoint,
    save_checkpoint,
    train,
)

TINY_MODEL_KWARGS = {
    "trunk_channels": 8,
    "num_residual_blocks": 1,
    "value_channels": 4,
    "value_hidden": 16,
}


def _generate_synthetic_pgn(num_games: int, seed: int = 0, max_plies: int = 16) -> str:
    """Random legal self-play games, just enough structure to exercise the loop."""
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


def _base_config(tmp_path: Path, **overrides) -> TrainingConfig:
    defaults = dict(
        dataset_path=_write_pgn(tmp_path),
        checkpoint_dir=tmp_path / "checkpoints",
        mlflow_tracking_uri=str(tmp_path / "mlruns"),
        batch_size=8,
        num_epochs=2,
        warmup_steps=2,
        num_workers=0,
        device="cpu",
        **TINY_MODEL_KWARGS,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


# --- MLflow URI normalization (Windows drive-letter footgun + file-store
# maintenance mode) --------------------------------------------------------


def test_resolve_mlflow_tracking_uri_converts_bare_path_to_sqlite(tmp_path: Path) -> None:
    windows_style = str(tmp_path / "mlruns")  # e.g. 'C:\\Users\\...\\mlruns' on Windows

    resolved = _resolve_mlflow_tracking_uri(windows_style)

    assert resolved.startswith("sqlite:///")
    expected_db_path = (Path(tmp_path) / "mlruns" / "mlflow.db").as_posix()
    assert resolved == f"sqlite:///{expected_db_path}"


def test_resolve_mlflow_tracking_uri_passes_through_real_uris() -> None:
    assert _resolve_mlflow_tracking_uri("http://localhost:5000") == "http://localhost:5000"
    assert _resolve_mlflow_tracking_uri("sqlite:///mlflow.db") == "sqlite:///mlflow.db"


# --- max_games cap, no training needed --------------------------------------


def test_effective_total_games_uncapped_returns_real_count(tmp_path: Path) -> None:
    path = _write_pgn(tmp_path, num_games=20)
    assert _effective_total_games(path, max_games=None) == 20


def test_effective_total_games_caps_at_max_games(tmp_path: Path) -> None:
    path = _write_pgn(tmp_path, num_games=20)
    assert _effective_total_games(path, max_games=5) == 5


def test_build_datasets_val_set_stops_at_capped_total_not_eof(tmp_path: Path) -> None:
    """Regression test for a real bug: an unbounded val_dataset (no end_game)
    reads to the physical end of the PGN file, not the logical end of a
    max_games-capped split. Uses a 40-game file with a simulated cap of 20
    (train_end=15) -- val should see games [15, 20), 5 games, not [15, 40),
    25 games."""
    path = _write_pgn(tmp_path, num_games=40)

    _train_dataset, val_dataset = _build_datasets(
        path, train_end=15, total_games=20, shuffle_buffer_size=0, seed=0
    )

    val_games_seen = sum(1 for _ in val_dataset._iter_own_games())
    assert val_games_seen == 5


def test_effective_total_games_cap_larger_than_file_is_a_no_op(tmp_path: Path) -> None:
    path = _write_pgn(tmp_path, num_games=20)
    assert _effective_total_games(path, max_games=1000) == 20


# --- LR schedule math, no training needed ----------------------------------


def test_warmup_cosine_lr_lambda_ramps_up_then_decays() -> None:
    warmup_steps, total_steps = 10, 100

    # Warmup: strictly increasing toward 1.0.
    warmup_values = [_warmup_cosine_lr_lambda(s, warmup_steps, total_steps) for s in range(10)]
    assert warmup_values == sorted(warmup_values)
    assert warmup_values[-1] <= 1.0

    # Just past warmup: near peak (cosine(0) = 1).
    assert _warmup_cosine_lr_lambda(warmup_steps, warmup_steps, total_steps) == pytest.approx(
        1.0, abs=1e-6
    )
    # At the end: cosine(pi) = -1 -> factor 0.
    assert _warmup_cosine_lr_lambda(total_steps, warmup_steps, total_steps) == pytest.approx(
        0.0, abs=1e-6
    )
    # Past the end: clamped, doesn't go negative or restart.
    assert _warmup_cosine_lr_lambda(total_steps + 50, warmup_steps, total_steps) == pytest.approx(
        0.0, abs=1e-6
    )


def test_estimate_total_steps_is_positive_and_scales_with_epochs(tmp_path: Path) -> None:
    config = _base_config(tmp_path, num_epochs=4)
    steps_4_epochs = _estimate_total_steps(train_games=15, config=config)

    config_2 = _base_config(tmp_path, num_epochs=2, dataset_path=config.dataset_path)
    steps_2_epochs = _estimate_total_steps(train_games=15, config=config_2)

    assert steps_4_epochs > 0
    assert steps_4_epochs == steps_2_epochs * 2


# --- checkpoint save/load, no training needed -------------------------------


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    config = _base_config(tmp_path)
    model = ChessNet(**TINY_MODEL_KWARGS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model,
        optimizer,
        scheduler,
        epoch=3,
        global_step=42,
        best_val_loss=0.5,
        config=config,
    )

    restored_model = ChessNet(**TINY_MODEL_KWARGS)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lr_lambda=lambda s: 1.0
    )

    checkpoint = load_checkpoint(path, restored_model, restored_optimizer, restored_scheduler)

    assert checkpoint["epoch"] == 3
    assert checkpoint["global_step"] == 42
    assert checkpoint["best_val_loss"] == 0.5
    for original, restored in zip(model.parameters(), restored_model.parameters(), strict=True):
        assert torch.equal(original, restored)


# --- end-to-end runs --------------------------------------------------------


def test_train_produces_checkpoints_and_mlflow_run(tmp_path: Path) -> None:
    config = _base_config(tmp_path)

    best_path = train(config)

    assert best_path.exists()
    assert (config.checkpoint_dir / "latest.pt").exists()

    checkpoint = torch.load(best_path, weights_only=False)
    assert checkpoint["epoch"] >= 0
    assert "model_state_dict" in checkpoint

    resolved_uri = _resolve_mlflow_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_uri)
    experiment = client.get_experiment_by_name(config.experiment_name)
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert "val_loss" in runs[0].data.metrics
    assert "val_policy_top1_accuracy" in runs[0].data.metrics

    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    assert len(versions) == 1
    assert versions[0].run_id == runs[0].info.run_id


def test_resume_from_checkpoint_continues_global_step(tmp_path: Path) -> None:
    config = _base_config(tmp_path, num_epochs=1)
    best_path = train(config)
    first_run_checkpoint = torch.load(best_path, weights_only=False)

    resumed_config = _base_config(
        tmp_path,
        dataset_path=config.dataset_path,
        checkpoint_dir=tmp_path / "checkpoints2",
        num_epochs=2,
        resume_from=best_path,
    )
    second_best_path = train(resumed_config)
    second_run_checkpoint = torch.load(second_best_path, weights_only=False)

    # Resumed run picks up epoch/global_step where the first run left off,
    # not from zero.
    assert second_run_checkpoint["epoch"] > first_run_checkpoint["epoch"]
    assert second_run_checkpoint["global_step"] > first_run_checkpoint["global_step"]
