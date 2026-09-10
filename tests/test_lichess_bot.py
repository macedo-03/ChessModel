"""Tests for the Lichess bot's logic: variant filtering, board reconstruction,
the per-gameState decision (move / wait / game over), and event dispatch --
all against a mocked LichessClient/API/Redis, no real network involved.
"""

from unittest.mock import MagicMock

import chess
import pytest

from chessmodel.bot.lichess_bot import (
    FINISHED_GAMES_QUEUE_KEY,
    BotClients,
    advance_game,
    handle_events,
    play_game,
    reconstruct_board,
    should_accept,
)

# --- should_accept ------------------------------------------------------


def test_should_accept_standard_variant() -> None:
    assert should_accept({"variant": {"key": "standard"}}) is True


@pytest.mark.parametrize("variant_key", ["chess960", "atomic", "kingOfTheHill"])
def test_should_accept_rejects_non_standard_variants(variant_key: str) -> None:
    assert should_accept({"variant": {"key": variant_key}}) is False


def test_should_accept_missing_variant_key() -> None:
    assert should_accept({}) is False


# --- reconstruct_board ---------------------------------------------------


def test_reconstruct_board_from_startpos() -> None:
    board = reconstruct_board("startpos", "e2e4 e7e5 g1f3")

    assert board.fen().startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2")
    assert board.turn == chess.BLACK


def test_reconstruct_board_no_moves_is_starting_position() -> None:
    board = reconstruct_board(None, "")

    assert board == chess.Board()


# --- advance_game ---------------------------------------------------------


def _make_clients() -> BotClients:
    lichess = MagicMock()
    api = MagicMock()
    api.post.return_value.json.return_value = {"move_uci": "e2e4"}
    redis_client = MagicMock()
    return BotClients(lichess=lichess, api=api, redis=redis_client, our_username="chessmodel-bot")


def test_advance_game_does_nothing_when_not_our_turn() -> None:
    clients = _make_clients()
    state = {"status": "started", "moves": "", "wtime": 60000, "btime": 60000}

    still_going = advance_game(clients, "g1", chess.BLACK, "startpos", state)

    assert still_going is True
    clients.api.post.assert_not_called()
    clients.lichess.make_move.assert_not_called()


def test_advance_game_submits_a_move_on_our_turn() -> None:
    clients = _make_clients()
    state = {"status": "started", "moves": "", "wtime": 60000, "btime": 60000}

    still_going = advance_game(clients, "g1", chess.WHITE, "startpos", state)

    assert still_going is True
    clients.api.post.assert_called_once_with("/move", json={"fen": chess.Board().fen()})
    clients.lichess.make_move.assert_called_once_with("g1", "e2e4")


def test_advance_game_pushes_finished_game_on_game_over() -> None:
    clients = _make_clients()
    clients.lichess.export_pgn.return_value = "1. e4 e5 *"
    state = {"status": "mate", "moves": "e2e4", "wtime": 60000, "btime": 60000}

    still_going = advance_game(clients, "g1", chess.WHITE, "startpos", state)

    assert still_going is False
    clients.api.post.assert_not_called()
    clients.lichess.export_pgn.assert_called_once_with("g1")
    clients.redis.lpush.assert_called_once()
    key, payload = clients.redis.lpush.call_args[0]
    assert key == FINISHED_GAMES_QUEUE_KEY
    assert "g1" in payload
    assert "1. e4 e5" in payload


# --- play_game (end-to-end against a mocked stream) -----------------------


def test_play_game_plays_our_moves_and_stops_at_game_over() -> None:
    clients = _make_clients()
    clients.lichess.export_pgn.return_value = "1. e4 *"
    clients.lichess.stream_game.return_value = iter(
        [
            {
                "type": "gameFull",
                "white": {"id": "chessmodel-bot"},
                "black": {"id": "opponent"},
                "initialFen": "startpos",
                "state": {"status": "started", "moves": "", "wtime": 60000, "btime": 60000},
            },
            {
                "type": "gameState",
                "status": "started",
                "moves": "e2e4",
                "wtime": 59000,
                "btime": 60000,
            },
            {
                "type": "gameState",
                "status": "mate",
                "moves": "e2e4",
                "wtime": 59000,
                "btime": 60000,
            },
        ]
    )

    play_game(clients, "g1")

    # We're white and it's our move at the initial state (moves="") -- one
    # call from that, none from the second event (Black's reply, not our
    # turn), and the game-over push from the third.
    assert clients.lichess.make_move.call_count == 1
    clients.redis.lpush.assert_called_once()


def test_play_game_detects_white_color_despite_id_casing_mismatch() -> None:
    # Regression test: Lichess's "id" field is always lowercase (the
    # canonical account id), while our_username preserves registered display
    # casing -- a case-sensitive comparison here silently always resolved to
    # BLACK, so the bot never moved first when actually assigned White.
    clients = _make_clients()
    clients.our_username = "XAIKnightmare"
    clients.lichess.stream_game.return_value = iter(
        [
            {
                "type": "gameFull",
                "white": {"id": "xaiknightmare"},
                "black": {"id": "opponent"},
                "initialFen": "startpos",
                "state": {"status": "started", "moves": "", "wtime": 60000, "btime": 60000},
            }
        ]
    )

    play_game(clients, "g1")

    clients.lichess.make_move.assert_called_once_with("g1", "e2e4")


# --- handle_events challenge dispatch --------------------------------------


def test_handle_events_accepts_standard_challenge() -> None:
    clients = _make_clients()
    clients.lichess.stream_events.return_value = iter(
        [{"type": "challenge", "challenge": {"id": "c1", "variant": {"key": "standard"}}}]
    )

    handle_events(clients)

    clients.lichess.accept_challenge.assert_called_once_with("c1")
    clients.lichess.decline_challenge.assert_not_called()


def test_handle_events_declines_non_standard_challenge() -> None:
    clients = _make_clients()
    clients.lichess.stream_events.return_value = iter(
        [{"type": "challenge", "challenge": {"id": "c1", "variant": {"key": "chess960"}}}]
    )

    handle_events(clients)

    clients.lichess.decline_challenge.assert_called_once_with("c1", reason="standard")
    clients.lichess.accept_challenge.assert_not_called()


def test_handle_events_survives_a_failed_challenge_response() -> None:
    # Regression test: accept_challenge raising (e.g. the challenge expired
    # between being seen and us responding) must not crash the whole event
    # loop -- it previously did, taking the entire bot process down.
    clients = _make_clients()
    clients.lichess.accept_challenge.side_effect = RuntimeError("404 Not Found")
    clients.lichess.stream_events.return_value = iter(
        [
            {"type": "challenge", "challenge": {"id": "c1", "variant": {"key": "standard"}}},
            {"type": "challenge", "challenge": {"id": "c2", "variant": {"key": "standard"}}},
        ]
    )

    handle_events(clients)  # must not raise

    assert clients.lichess.accept_challenge.call_count == 2
