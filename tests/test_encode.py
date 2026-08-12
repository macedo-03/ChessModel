"""Tests for board/move/value encoding (ADR-0001, ADR-0003, ADR-0002 fix)."""

import chess
import numpy as np

from chessmodel.data.encode import (
    NUM_INPUT_PLANES,
    NUM_POLICY_PLANES,
    board_to_tensor,
    move_to_policy_index,
    policy_index_to_move,
    value_target,
)

# --- board_to_tensor (ADR-0001) ---------------------------------------------


def test_board_to_tensor_shape_and_dtype() -> None:
    tensor = board_to_tensor(chess.Board())
    assert tensor.shape == (NUM_INPUT_PLANES, 8, 8)
    assert tensor.dtype == np.float32


def test_board_to_tensor_starting_position_piece_planes() -> None:
    tensor = board_to_tensor(chess.Board())

    # White pawns: plane 0, rank 2 -> row 6, all columns.
    assert np.all(tensor[0, 6, :] == 1.0)
    # Black pawns: plane 6, rank 7 -> row 1.
    assert np.all(tensor[6, 1, :] == 1.0)
    # White king: plane 5, e1 -> row 7, col 4.
    assert tensor[5, 7, 4] == 1.0
    # Black king: plane 11, e8 -> row 0, col 4.
    assert tensor[11, 0, 4] == 1.0
    # Empty ranks (3-6) are all zero across every piece plane.
    assert np.all(tensor[0:12, 2:6, :] == 0.0)


def test_board_to_tensor_starting_position_broadcast_planes() -> None:
    tensor = board_to_tensor(chess.Board())

    assert np.all(tensor[12, :, :] == 1.0)  # White to move
    assert np.all(tensor[13, :, :] == 1.0)  # White kingside castling
    assert np.all(tensor[14, :, :] == 1.0)  # White queenside castling
    assert np.all(tensor[15, :, :] == 1.0)  # Black kingside castling
    assert np.all(tensor[16, :, :] == 1.0)  # Black queenside castling
    assert np.all(tensor[17, :, :] == 0.0)  # No en passant target
    assert np.all(tensor[18, :, :] == 0.0)  # Halfmove clock is 0


def test_board_to_tensor_side_to_move_flips() -> None:
    board = chess.Board()
    board.push_san("e4")
    tensor = board_to_tensor(board)
    assert np.all(tensor[12, :, :] == 0.0)  # Black to move


def test_board_to_tensor_en_passant_target_square() -> None:
    board = chess.Board()
    board.push_san("e4")  # e2-e4 opens the e3 en passant square for Black... actually for White's
    board.push_san("Nf6")
    board.push_san("e5")
    board.push_san("d5")  # d7-d5, en passant target is d6
    tensor = board_to_tensor(board)

    d6 = chess.parse_square("d6")
    row, col = 7 - chess.square_rank(d6), chess.square_file(d6)
    assert tensor[17, row, col] == 1.0
    assert tensor[17].sum() == 1.0


def test_board_to_tensor_castling_rights_cleared_after_king_move() -> None:
    board = chess.Board()
    for san in ["e4", "e5", "Ke2", "Ke7"]:
        board.push_san(san)
    tensor = board_to_tensor(board)

    assert np.all(tensor[13, :, :] == 0.0)
    assert np.all(tensor[14, :, :] == 0.0)
    assert np.all(tensor[15, :, :] == 0.0)
    assert np.all(tensor[16, :, :] == 0.0)


def test_board_to_tensor_halfmove_clock_normalization() -> None:
    board = chess.Board()
    board.halfmove_clock = 50
    tensor = board_to_tensor(board)
    assert np.all(tensor[18, :, :] == 0.5)

    board.halfmove_clock = 250  # beyond the 100 cap
    tensor = board_to_tensor(board)
    assert np.all(tensor[18, :, :] == 1.0)


# --- move_to_policy_index (ADR-0003) ----------------------------------------
# Each expected value here matches a worked example already walked through by
# hand — if these ever drift, the hand-computed explanation is wrong too.


def test_move_to_policy_index_pawn_push_e2e4() -> None:
    move = chess.Move.from_uci("e2e4")
    assert move_to_policy_index(move) == 3797


def test_move_to_policy_index_knight_jump_nb1c3() -> None:
    move = chess.Move.from_uci("b1c3")
    assert move_to_policy_index(move) == 4218


def test_move_to_policy_index_bishop_slide_bf1b5() -> None:
    move = chess.Move.from_uci("f1b5")
    assert move_to_policy_index(move) == 4505


def test_move_to_policy_index_castling_e1g1() -> None:
    move = chess.Move.from_uci("e1g1")
    assert move_to_policy_index(move) == 4395


def test_move_to_policy_index_auto_queen_promotion() -> None:
    move = chess.Move.from_uci("e7e8q")
    assert move_to_policy_index(move) == 876


def test_move_to_policy_index_underpromotion_to_knight_same_squares_different_index() -> None:
    move = chess.Move.from_uci("e7e8n")
    assert move_to_policy_index(move) == 943


def test_move_to_policy_index_capturing_underpromotion_to_rook() -> None:
    move = chess.Move.from_uci("d7e8r")
    assert move_to_policy_index(move) == 875


def test_policy_index_bounds() -> None:
    # Every plane index used above must land inside the documented 0-4671 range.
    for uci in ["e2e4", "b1c3", "f1b5", "e1g1", "e7e8q", "e7e8n", "d7e8r"]:
        index = move_to_policy_index(chess.Move.from_uci(uci))
        assert 0 <= index < 8 * 8 * NUM_POLICY_PLANES


# --- round-trip: move_to_policy_index <-> policy_index_to_move -------------


def _assert_all_legal_moves_roundtrip(board: chess.Board) -> None:
    for move in board.legal_moves:
        index = move_to_policy_index(move)
        decoded = policy_index_to_move(index, board)
        assert decoded == move, f"roundtrip failed for {move} in {board.fen()}"


def test_roundtrip_starting_position() -> None:
    _assert_all_legal_moves_roundtrip(chess.Board())


def test_roundtrip_across_a_played_game() -> None:
    board = chess.Board()
    for san in ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"]:
        _assert_all_legal_moves_roundtrip(board)
        board.push_san(san)
    _assert_all_legal_moves_roundtrip(board)


def test_roundtrip_covers_push_and_capture_promotions() -> None:
    # White pawn on d7 can push to d8 or capture the rook on c8, each with
    # all four promotion choices (Q/R/B/N) — exercises both the queen-type
    # plane branch (auto-queen) and the underpromotion branch, in both the
    # "straight" and "capture" directions.
    board = chess.Board("2r1k3/3P4/8/8/8/8/8/4K3 w - - 0 1")
    _assert_all_legal_moves_roundtrip(board)


# --- value_target (ADR-0002 fix) --------------------------------------------


def test_value_target_white_perspective() -> None:
    assert value_target("1-0", white_to_move=True) == 1.0
    assert value_target("0-1", white_to_move=True) == -1.0
    assert value_target("1/2-1/2", white_to_move=True) == 0.0


def test_value_target_black_perspective() -> None:
    assert value_target("1-0", white_to_move=False) == -1.0
    assert value_target("0-1", white_to_move=False) == 1.0
    assert value_target("1/2-1/2", white_to_move=False) == 0.0


def test_value_target_same_game_differs_by_mover() -> None:
    # The exact bug ADR-0002 flags: the same game result must NOT produce the
    # same label regardless of whose turn it was at a given position.
    result = "1-0"
    assert value_target(result, white_to_move=True) != value_target(result, white_to_move=False)
