"""Tests for the streaming position Dataset (ChessPositionDataset et al.)."""

from pathlib import Path
from types import SimpleNamespace

import chess
import torch

from chessmodel.data.dataset import (
    ChessPositionDataset,
    collate_samples,
    count_games,
    split_train_val,
)
from chessmodel.data.encode import (
    NUM_INPUT_PLANES,
    board_to_tensor,
    move_to_policy_index,
    value_target,
)

# Game 1: White wins. Game 2: Black wins. Game 3: unfinished (no usable label).
SYNTHETIC_PGN = """\
[Event "Test"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Nf3 1-0

[Event "Test"]
[White "C"]
[Black "D"]
[Result "0-1"]

1. d4 d5 0-1

[Event "Test"]
[White "E"]
[Black "F"]
[Result "*"]

1. c4 *
"""


def _write_pgn(tmp_path: Path, text: str = SYNTHETIC_PGN) -> Path:
    path = tmp_path / "games.pgn"
    path.write_text(text, encoding="utf-8")
    return path


def test_count_games_counts_every_block_including_unfinished(tmp_path: Path) -> None:
    path = _write_pgn(tmp_path)
    assert count_games(path) == 3


def test_split_train_val_boundary() -> None:
    train_end, total = split_train_val(1000, val_fraction=0.05)
    assert total == 1000
    assert train_end == 950


def test_split_train_val_always_keeps_at_least_one_val_game() -> None:
    train_end, total = split_train_val(5, val_fraction=0.01)
    assert total == 5
    assert train_end == 4  # at least 1 val game, even when the fraction rounds to 0


def test_dataset_yields_expected_positions_in_order(tmp_path: Path) -> None:
    path = _write_pgn(tmp_path)
    dataset = ChessPositionDataset(path, shuffle_buffer_size=0)
    samples = list(dataset)

    # 3 plies from game 1 + 2 plies from game 2; game 3 is unfinished, contributes 0.
    assert len(samples) == 5

    board = chess.Board()
    expected_moves = ["e4", "e5", "Nf3"]
    for (tensor, policy_index, value), san in zip(samples[:3], expected_moves, strict=True):
        move = board.parse_san(san)
        np_expected = board_to_tensor(board)
        assert (tensor == np_expected).all()
        assert policy_index == move_to_policy_index(move)
        assert value == value_target("1-0", white_to_move=board.turn == chess.WHITE)
        board.push(move)

    board = chess.Board()
    for (tensor, policy_index, value), san in zip(samples[3:], ["d4", "d5"], strict=True):
        move = board.parse_san(san)
        np_expected = board_to_tensor(board)
        assert (tensor == np_expected).all()
        assert policy_index == move_to_policy_index(move)
        assert value == value_target("0-1", white_to_move=board.turn == chess.WHITE)
        board.push(move)


def test_dataset_end_game_excludes_later_games(tmp_path: Path) -> None:
    path = _write_pgn(tmp_path)
    dataset = ChessPositionDataset(path, end_game=1, shuffle_buffer_size=0)
    samples = list(dataset)
    assert len(samples) == 3  # only game 1's three plies


def test_dataset_start_game_excludes_earlier_games(tmp_path: Path) -> None:
    path = _write_pgn(tmp_path)
    dataset = ChessPositionDataset(path, start_game=1, shuffle_buffer_size=0)
    samples = list(dataset)
    assert len(samples) == 2  # only game 2's two plies (game 3 unfinished)


def test_dataset_shuffle_buffer_preserves_full_sample_set(tmp_path: Path) -> None:
    path = _write_pgn(tmp_path)
    unshuffled = list(ChessPositionDataset(path, shuffle_buffer_size=0))
    shuffled = list(ChessPositionDataset(path, shuffle_buffer_size=2, seed=42))

    assert len(shuffled) == len(unshuffled)
    assert sorted(s[1] for s in shuffled) == sorted(s[1] for s in unshuffled)


def test_dataset_shards_games_across_workers(monkeypatch, tmp_path: Path) -> None:
    path = _write_pgn(tmp_path)

    monkeypatch.setattr(
        "chessmodel.data.dataset.get_worker_info",
        lambda: SimpleNamespace(id=0, num_workers=2),
    )
    worker0_samples = list(ChessPositionDataset(path, shuffle_buffer_size=0))

    monkeypatch.setattr(
        "chessmodel.data.dataset.get_worker_info",
        lambda: SimpleNamespace(id=1, num_workers=2),
    )
    worker1_samples = list(ChessPositionDataset(path, shuffle_buffer_size=0))

    # Game 0 (even index) goes to worker 0, game 1 (odd index) to worker 1,
    # game 2 is unfinished either way. No overlap, and together they cover
    # every position the unsharded run produces.
    assert len(worker0_samples) == 3  # game 1's three plies
    assert len(worker1_samples) == 2  # game 2's two plies

    monkeypatch.undo()
    full = list(ChessPositionDataset(path, shuffle_buffer_size=0))
    assert len(worker0_samples) + len(worker1_samples) == len(full)


def test_collate_samples_shapes_and_dtypes(tmp_path: Path) -> None:
    path = _write_pgn(tmp_path)
    batch = list(ChessPositionDataset(path, shuffle_buffer_size=0))[:2]

    boards, policies, values = collate_samples(batch)

    assert boards.shape == (2, NUM_INPUT_PLANES, 8, 8)
    assert boards.dtype == torch.float32
    assert policies.shape == (2,)
    assert policies.dtype == torch.long
    assert values.shape == (2,)
    assert values.dtype == torch.float32
