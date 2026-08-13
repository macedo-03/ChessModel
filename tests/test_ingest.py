"""Tests for the PGN ingestion filter logic.

No network access — filter_games is exercised against a small synthetic PGN
blob via io.StringIO, never the real Lichess dump.
"""

import io

import chess.pgn

from chessmodel.data.ingest import (
    build_dump_url,
    filter_games,
    game_qualifies,
    is_valid_time_control,
    read_next_game_block,
)

SYNTHETIC_PGN = """\
[Event "Rated Blitz game"]
[White "PlayerA"]
[Black "PlayerB"]
[Result "1-0"]
[WhiteElo "2600"]
[BlackElo "2550"]
[TimeControl "600+5"]

1. e4 e5 2. Nf3 1-0

[Event "Rated Blitz game"]
[White "PlayerC"]
[Black "PlayerD"]
[Result "0-1"]
[WhiteElo "2100"]
[BlackElo "2600"]
[TimeControl "600+5"]

1. d4 d5 2. c4 0-1

[Event "Rated Bullet game"]
[White "PlayerE"]
[Black "PlayerF"]
[Result "1-0"]
[WhiteElo "2700"]
[BlackElo "2700"]
[TimeControl "60+0"]

1. e4 e5 1-0

[Event "Rated Rapid game"]
[White "PlayerG"]
[Black "PlayerH"]
[Result "1/2-1/2"]
[WhiteElo "2500"]
[BlackElo "2500"]
[TimeControl "180+0"]

1. e4 e5 1/2-1/2
"""


def test_build_dump_url() -> None:
    assert build_dump_url("2026-07") == (
        "https://database.lichess.org/standard/lichess_db_standard_rated_2026-07.pgn.zst"
    )


def test_is_valid_time_control_above_and_below_threshold() -> None:
    assert is_valid_time_control("600+5", min_seconds=180)
    assert is_valid_time_control("180+0", min_seconds=180)
    assert not is_valid_time_control("60+0", min_seconds=180)


def test_is_valid_time_control_malformed_is_rejected() -> None:
    assert not is_valid_time_control("-", min_seconds=180)
    assert not is_valid_time_control("", min_seconds=180)


def test_game_qualifies_requires_both_players_above_elo() -> None:
    # game_qualifies takes a Mapping[str, str] -- a plain dict works fine,
    # not just chess.pgn.Headers, since ingestion no longer builds one.
    headers = {"WhiteElo": "2100", "BlackElo": "2600", "TimeControl": "600+5"}
    assert not game_qualifies(headers, min_elo=2500, min_time_control_seconds=180)

    headers["WhiteElo"] = "2600"
    assert game_qualifies(headers, min_elo=2500, min_time_control_seconds=180)


def test_game_qualifies_rejects_bullet() -> None:
    headers = {"WhiteElo": "2700", "BlackElo": "2700", "TimeControl": "60+0"}
    assert not game_qualifies(headers, min_elo=2500, min_time_control_seconds=180)


def test_game_qualifies_missing_elo_is_rejected() -> None:
    headers = {"TimeControl": "600+5"}
    assert not game_qualifies(headers, min_elo=2500, min_time_control_seconds=180)


FIRST_GAME_BLOCK = (
    '[Event "Rated Blitz game"]\n'
    '[White "PlayerA"]\n'
    '[Black "PlayerB"]\n'
    '[Result "1-0"]\n'
    '[WhiteElo "2600"]\n'
    '[BlackElo "2550"]\n'
    '[TimeControl "600+5"]\n'
    "\n"
    "1. e4 e5 2. Nf3 1-0\n"
    "\n"
)


def test_read_next_game_block_parses_headers_and_returns_none_at_eof() -> None:
    stream = io.StringIO(SYNTHETIC_PGN)

    headers, lines = read_next_game_block(stream)
    assert headers["White"] == "PlayerA"
    assert headers["WhiteElo"] == "2600"
    assert "".join(lines) == FIRST_GAME_BLOCK

    # Consume the remaining three games...
    for _ in range(3):
        assert read_next_game_block(stream) is not None
    # ...and EOF is reported cleanly.
    assert read_next_game_block(stream) is None


def test_filter_games_output_is_byte_identical_for_kept_games() -> None:
    # Filtering only ever reads headers -- a kept game's text must pass
    # through completely unchanged, never reconstructed via python-chess.
    source = io.StringIO(SYNTHETIC_PGN)
    output = io.StringIO()

    filter_games(source, output, min_elo=2500, min_time_control_seconds=180, max_games=None)

    assert output.getvalue().startswith(FIRST_GAME_BLOCK)


def test_filter_games_keeps_only_qualifying_games() -> None:
    source = io.StringIO(SYNTHETIC_PGN)
    output = io.StringIO()

    scanned, kept = filter_games(
        source, output, min_elo=2500, min_time_control_seconds=180, max_games=None
    )

    assert scanned == 4
    assert kept == 2

    output.seek(0)
    kept_games = []
    while (game := chess.pgn.read_game(output)) is not None:
        kept_games.append(game)

    assert len(kept_games) == 2
    assert {g.headers["White"] for g in kept_games} == {"PlayerA", "PlayerG"}


def test_filter_games_respects_max_games_cap() -> None:
    source = io.StringIO(SYNTHETIC_PGN)
    output = io.StringIO()

    scanned, kept = filter_games(
        source, output, min_elo=2500, min_time_control_seconds=180, max_games=1
    )

    assert kept == 1
    # Stops as soon as the cap is hit, without scanning the rest of the stream.
    assert scanned == 1


def test_filter_games_respects_max_scanned_cap_independent_of_hit_rate() -> None:
    # min_elo=9999 means nothing ever qualifies — max_scanned is the only thing
    # that can stop the loop, exactly the scenario a strict rating filter creates.
    source = io.StringIO(SYNTHETIC_PGN)
    output = io.StringIO()

    scanned, kept = filter_games(
        source, output, min_elo=9999, min_time_control_seconds=180, max_scanned=2
    )

    assert scanned == 2
    assert kept == 0
