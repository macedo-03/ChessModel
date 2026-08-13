"""Tests for the dataset card generator, against a small synthetic PGN."""

from pathlib import Path

from chessmodel.data.dataset_card import collect_header_stats, estimate_avg_plies, render_card

SYNTHETIC_PGN = """\
[Event "Test"]
[White "A"]
[Black "B"]
[Result "1-0"]
[UTCDate "2026.07.01"]
[WhiteElo "2600"]
[BlackElo "2550"]
[TimeControl "600+5"]

1. e4 e5 2. Nf3 1-0

[Event "Test"]
[White "C"]
[Black "D"]
[Result "0-1"]
[UTCDate "2026.07.15"]
[WhiteElo "2300"]
[BlackElo "2700"]
[TimeControl "600+5"]

1. d4 d5 0-1

[Event "Test"]
[White "E"]
[Black "F"]
[Result "1/2-1/2"]
[UTCDate "2026.07.10"]
[WhiteElo "2400"]
[BlackElo "2400"]
[TimeControl "300+0"]

1. c4 c5 2. Nf3 Nf6 1/2-1/2

[Event "Test"]
[White "G"]
[Black "H"]
[Result "*"]
[UTCDate "2026.07.20"]
[WhiteElo "2250"]
[BlackElo "2250"]
[TimeControl "180+0"]

1. Nf3 *
"""


def _write_pgn(tmp_path: Path) -> Path:
    path = tmp_path / "games.pgn"
    path.write_text(SYNTHETIC_PGN, encoding="utf-8")
    return path


def test_collect_header_stats(tmp_path: Path) -> None:
    stats = collect_header_stats(_write_pgn(tmp_path))

    assert stats["game_count"] == 4
    assert stats["result_counts"]["1-0"] == 1
    assert stats["result_counts"]["0-1"] == 1
    assert stats["result_counts"]["1/2-1/2"] == 1
    assert stats["result_counts"]["*"] == 1
    assert sorted(stats["elos"]) == [2250, 2250, 2300, 2400, 2400, 2550, 2600, 2700]
    assert sorted(stats["dates"]) == ["2026.07.01", "2026.07.10", "2026.07.15", "2026.07.20"]
    assert stats["time_control_valid_count"] == 4


def test_estimate_avg_plies_only_samples_finished_games(tmp_path: Path) -> None:
    avg, sampled = estimate_avg_plies(_write_pgn(tmp_path), sample_size=10)

    # 3 finished games (3, 2, and 4 plies) -- the unfinished 4th game is excluded.
    assert sampled == 3
    assert avg == (3 + 2 + 4) / 3


def test_render_card_contains_real_numbers(tmp_path: Path) -> None:
    path = _write_pgn(tmp_path)
    card = render_card(
        path,
        dvc_md5="deadbeef",
        source_month="2026-07",
        min_elo=2200,
        min_time_control_seconds=180,
        generated_date="2026-08-12",
        sample_size=10,
    )

    assert "Games | 4" in card
    assert "White wins | 1 " in card
    assert "Black wins | 1 " in card
    assert "Draws | 1 " in card
    assert "Unfinished" in card
    assert "2026.07.01 to 2026.07.20" in card
    assert "deadbeef" in card
    assert "2200" in card
