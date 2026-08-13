"""PyTorch Dataset over encoded chess positions, streamed from a filtered PGN file.

Games are replayed once, sequentially, front-to-back -- each position can only
be encoded after replaying every prior move in that game (whose-turn, castling
rights, etc. depend on move history), so this is fundamentally a streaming
scan, not a randomly-indexable dataset: seeking to an arbitrary position would
mean replaying every move before it anyway. Positions are shuffled via a
bounded in-memory buffer rather than true random access, which avoids both a
full materialization pass (ADR-0002) and the wasted, repeated replay a naive
random-access index would cost.

Filtering (ingest.py) never touches python-chess, since it only needs PGN
headers. This module is the other half: it's where chess-aware processing
actually happens, but only for the small, already-filtered subset of games
ingest.py kept -- never the full firehose of a monthly dump.
"""

from __future__ import annotations

import io
import random
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from chessmodel.data.encode import board_to_tensor, move_to_policy_index, value_target
from chessmodel.data.ingest import read_next_game_block

Sample = tuple[np.ndarray, int, float]

DEFAULT_VAL_FRACTION = 0.05
DEFAULT_SHUFFLE_BUFFER_SIZE = 10_000

_FINISHED_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def count_games(pgn_path: Path) -> int:
    """Count games in a filtered PGN file via the fast text scanner (no chess parsing)."""
    count = 0
    with pgn_path.open("r", encoding="utf-8") as stream:
        while read_next_game_block(stream) is not None:
            count += 1
    return count


def split_train_val(
    total_games: int, val_fraction: float = DEFAULT_VAL_FRACTION
) -> tuple[int, int]:
    """Chronological split boundary: the first (1 - val_fraction) games are train.

    The source file is already in chronological order (games kept in the order
    ingest.py scanned them), so a game-index boundary is a time boundary too --
    Phase 01's "split by game, chronologically" requirement, satisfied without
    reshuffling anything.

    Returns (train_end, total_games) as exclusive upper bounds, directly usable
    as ChessPositionDataset(..., end_game=train_end) for train and
    ChessPositionDataset(..., start_game=train_end) for validation.
    """
    val_games = max(1, int(total_games * val_fraction))
    train_end = total_games - val_games
    return train_end, total_games


def _iter_game_texts(stream: TextIO) -> Iterator[str]:
    """Yield each game's raw PGN text (headers + movetext) from stream, in order."""
    while True:
        block = read_next_game_block(stream)
        if block is None:
            return
        _headers, lines = block
        yield "".join(lines)


def _positions_from_game(game_text: str) -> Iterator[Sample]:
    """Replay one game's moves, yielding an encoded (board, policy, value) triple per ply."""
    game = chess.pgn.read_game(io.StringIO(game_text))
    if game is None:
        return

    result = game.headers.get("Result", "*")
    if result not in _FINISHED_RESULTS:
        return  # unfinished/aborted game -- no usable outcome label

    board = game.board()
    for move in game.mainline_moves():
        tensor = board_to_tensor(board)
        policy_index = move_to_policy_index(move)
        value = value_target(result, white_to_move=board.turn == chess.WHITE)
        yield tensor, policy_index, value
        board.push(move)


class ChessPositionDataset(IterableDataset[Sample]):
    """Streams (board_tensor, policy_index, value) triples from a filtered PGN file.

    start_game/end_game select a contiguous, chronologically-ordered slice of
    games (see split_train_val for picking the boundary between train/val).
    shuffle_buffer_size trades memory for shuffle quality: 0 disables
    shuffling entirely (strict game order); larger values approximate random
    order without ever materializing the full dataset in memory.

    Safe to use with DataLoader(num_workers>1): games are sharded across
    workers by index, so each position is yielded by exactly one worker.
    """

    def __init__(
        self,
        pgn_path: Path,
        start_game: int = 0,
        end_game: int | None = None,
        shuffle_buffer_size: int = DEFAULT_SHUFFLE_BUFFER_SIZE,
        seed: int = 0,
    ) -> None:
        self.pgn_path = pgn_path
        self.start_game = start_game
        self.end_game = end_game
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def _iter_own_games(self) -> Iterator[str]:
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        with self.pgn_path.open("r", encoding="utf-8") as stream:
            for index, game_text in enumerate(_iter_game_texts(stream)):
                if index < self.start_game:
                    continue
                if self.end_game is not None and index >= self.end_game:
                    break
                if index % num_workers == worker_id:
                    yield game_text

    def _iter_positions(self) -> Iterator[Sample]:
        for game_text in self._iter_own_games():
            yield from _positions_from_game(game_text)

    def __iter__(self) -> Iterator[Sample]:
        if self.shuffle_buffer_size <= 0:
            yield from self._iter_positions()
            return

        rng = random.Random(self.seed)
        buffer: list[Sample] = []
        for sample in self._iter_positions():
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
                continue
            swap_index = rng.randrange(len(buffer))
            yield buffer[swap_index]
            buffer[swap_index] = sample
        rng.shuffle(buffer)
        yield from buffer


def collate_samples(batch: list[Sample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack a list of (board, policy_index, value) samples into batched tensors.

    Explicit rather than relying on DataLoader's default collate, which would
    infer dtypes (e.g. float64 for Python floats) that don't match what the
    training loop expects: float32 throughout, long for the CrossEntropyLoss
    policy target.
    """
    boards = torch.from_numpy(np.stack([sample[0] for sample in batch]))
    policies = torch.tensor([sample[1] for sample in batch], dtype=torch.long)
    values = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)
    return boards, policies, values
