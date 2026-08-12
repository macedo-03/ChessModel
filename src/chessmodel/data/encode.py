"""Position and move encoding.

Board tensor per ADR-0001 (docs/adr/0001-bitboard-tensor-spec.md), policy move
index per ADR-0003 (docs/adr/0003-policy-move-encoding.md), and the mover's-
perspective value target that fixes the bug documented in ADR-0002.
"""

from __future__ import annotations

import chess
import numpy as np

NUM_INPUT_PLANES = 19
NUM_POLICY_PLANES = 73
NUM_POLICY_OUTPUTS = 8 * 8 * NUM_POLICY_PLANES  # 4,672

_PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

# Index order fixes the numbering referenced throughout ADR-0003's worked examples.
_QUEEN_DIRECTIONS = [
    (-1, 0),  # 0: N
    (-1, 1),  # 1: NE
    (0, 1),  # 2: E
    (1, 1),  # 3: SE
    (1, 0),  # 4: S
    (1, -1),  # 5: SW
    (0, -1),  # 6: W
    (-1, -1),  # 7: NW
]
_KNIGHT_OFFSETS = [
    (-2, -1),
    (-2, 1),
    (-1, 2),
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
]
_UNDERPROMOTION_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _square_to_row_col(square: chess.Square) -> tuple[int, int]:
    """Row 0 = rank 8, column 0 = file a (ADR-0001)."""
    return 7 - chess.square_rank(square), chess.square_file(square)


def _row_col_to_square(row: int, col: int) -> chess.Square:
    return chess.square(col, 7 - row)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a position per ADR-0001.

    Returns a (19, 8, 8) float32 array — channels-first (plane, row, col),
    ready for Conv2d(in_channels=19, ...) with no further reshaping.
    """
    tensor = np.zeros((NUM_INPUT_PLANES, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():
        row, col = _square_to_row_col(square)
        plane = _PIECE_TYPES.index(piece.piece_type)
        if piece.color == chess.BLACK:
            plane += 6
        tensor[plane, row, col] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    if board.ep_square is not None:
        row, col = _square_to_row_col(board.ep_square)
        tensor[17, row, col] = 1.0

    tensor[18, :, :] = min(board.halfmove_clock, 100) / 100.0

    return tensor


def move_to_policy_index(move: chess.Move) -> int:
    """Encode a move per ADR-0003's 8x8x73 move-plane scheme."""
    from_row, from_col = _square_to_row_col(move.from_square)
    to_row, to_col = _square_to_row_col(move.to_square)
    d_row = to_row - from_row
    d_col = to_col - from_col

    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = _UNDERPROMOTION_PIECES.index(move.promotion)
        dir_idx = d_col + 1  # -1 (capture-left) -> 0, 0 (straight) -> 1, +1 (capture-right) -> 2
        plane = 64 + dir_idx * 3 + piece_idx
    elif d_row == 0 or d_col == 0 or abs(d_row) == abs(d_col):
        direction = ((d_row > 0) - (d_row < 0), (d_col > 0) - (d_col < 0))
        dir_idx = _QUEEN_DIRECTIONS.index(direction)
        distance = max(abs(d_row), abs(d_col))
        plane = dir_idx * 7 + (distance - 1)
    else:
        plane = 56 + _KNIGHT_OFFSETS.index((d_row, d_col))

    origin_flat = from_row * 8 + from_col
    return origin_flat * NUM_POLICY_PLANES + plane


def policy_index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Decode a policy index back into a chess.Move.

    Needs board context only to resolve auto-queen promotion — deciding
    whether a queen-type-plane move that lands on the back rank is a
    promotion at all depends on whether a pawn is actually making it, which
    isn't recoverable from the index alone (see ADR-0003's rationale).

    Assumes index came from board.legal_moves (post-masking); does not bounds-
    check moves that would run off the board, since that scenario shouldn't
    occur on the actual inference/training path.
    """
    origin_flat, plane = divmod(index, NUM_POLICY_PLANES)
    from_row, from_col = divmod(origin_flat, 8)
    from_square = _row_col_to_square(from_row, from_col)
    moving_piece = board.piece_at(from_square)

    promotion: chess.PieceType | None = None

    if plane < 56:
        dir_idx, dist_idx = divmod(plane, 7)
        d_row, d_col = _QUEEN_DIRECTIONS[dir_idx]
        distance = dist_idx + 1
        to_row = from_row + d_row * distance
        to_col = from_col + d_col * distance
    elif plane < 64:
        d_row, d_col = _KNIGHT_OFFSETS[plane - 56]
        to_row = from_row + d_row
        to_col = from_col + d_col
    else:
        dir_idx, piece_idx = divmod(plane - 64, 3)
        d_col = dir_idx - 1
        d_row = -1 if moving_piece is not None and moving_piece.color == chess.WHITE else 1
        to_row = from_row + d_row
        to_col = from_col + d_col
        promotion = _UNDERPROMOTION_PIECES[piece_idx]

    to_square = _row_col_to_square(to_row, to_col)

    is_pawn = moving_piece is not None and moving_piece.piece_type == chess.PAWN
    if promotion is None and is_pawn and chess.square_rank(to_square) in (0, 7):
        promotion = chess.QUEEN

    return chess.Move(from_square, to_square, promotion=promotion)


def value_target(result: str, white_to_move: bool) -> float:
    """Game outcome from the mover's perspective — the ADR-0002 fix.

    result: PGN Result header — "1-0", "0-1", or "1/2-1/2".
    white_to_move: whose turn it is at the position this label is attached to
        (not whose turn it was at the end of the game).
    """
    if result == "1/2-1/2":
        return 0.0
    white_won = result == "1-0"
    mover_won = white_won if white_to_move else not white_won
    return 1.0 if mover_won else -1.0
