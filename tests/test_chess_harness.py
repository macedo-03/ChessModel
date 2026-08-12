"""Sanity harness for chess-rule handling via python-chess.

Fixes the correctness baseline before any encoder, model, or bot code is written
on top of it. Draw-condition coverage here exists specifically so Phase 5 (the
Lichess bot) doesn't discover repetition/50-move/insufficient-material bugs live.
"""

import io

import chess
import chess.pgn

KNOWN_PERFT = {
    1: 20,
    2: 400,
    3: 8_902,
}


def _perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    nodes = 0
    for move in board.legal_moves:
        board.push(move)
        nodes += _perft(board, depth - 1)
        board.pop()
    return nodes


def test_fen_roundtrip_starting_position() -> None:
    board = chess.Board()
    assert board.fen() == chess.STARTING_FEN


def test_fen_roundtrip_preserves_castling_and_en_passant() -> None:
    fen = "rnbqkbnr/pp1ppppp/8/2pP4/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3"
    board = chess.Board(fen)
    assert board.fen() == fen


def test_legal_move_count_starting_position() -> None:
    board = chess.Board()
    assert len(list(board.legal_moves)) == KNOWN_PERFT[1]


def test_perft_depth_two_and_three_starting_position() -> None:
    board = chess.Board()
    assert _perft(board, 2) == KNOWN_PERFT[2]
    assert _perft(board, 3) == KNOWN_PERFT[3]


def test_pgn_parsing_moves_are_legal_in_sequence() -> None:
    pgn_text = """
[Event "Test"]
[Site "?"]
[Date "????.??.??"]
[Round "?"]
[White "A"]
[Black "B"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 1-0
"""
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    assert game is not None

    board = game.board()
    move_count = 0
    for move in game.mainline_moves():
        assert move in board.legal_moves
        board.push(move)
        move_count += 1

    assert move_count == 10
    assert game.headers["Result"] == "1-0"


def test_threefold_repetition_is_detected() -> None:
    board = chess.Board()
    assert not board.can_claim_threefold_repetition()

    # Two full knight-shuffle cycles return to the starting position for the
    # 3rd time: initial occurrence, after cycle 1, after cycle 2.
    for san in ["Nf3", "Nf6", "Ng1", "Ng8"] * 2:
        board.push_san(san)

    assert board.can_claim_threefold_repetition()


def test_fifty_move_rule_is_detected() -> None:
    board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    board.halfmove_clock = 100
    assert board.can_claim_fifty_moves()


def test_insufficient_material_king_vs_king() -> None:
    board = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    assert board.is_insufficient_material()


def test_sufficient_material_with_extra_queen_is_not_flagged() -> None:
    board = chess.Board("8/8/8/4k3/8/8/4K1Q1/8 w - - 0 1")
    assert not board.is_insufficient_material()
