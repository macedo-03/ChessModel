"""Generate a dataset card: summary statistics and provenance for a filtered PGN file.

Two passes, both bounded to be cheap:

1. A header-only scan over every game (game count, result distribution, Elo
   range, date range, time-control distribution) -- fast, no chess parsing,
   same read_next_game_block scanner ingestion uses.
2. A reservoir-sampled subset of games gets a real python-chess replay to
   estimate average plies/game, extrapolated to a position-count estimate.
   Replaying all 200k games for an exact count would take ~15 minutes (the
   same per-game cost that made the original ingest.py slow) -- not worth
   paying just to produce documentation, so this is disclosed as an estimate,
   not presented as exact.
"""

from __future__ import annotations

import argparse
import io
import random
from collections import Counter
from pathlib import Path

import chess.pgn

from chessmodel.data.ingest import read_next_game_block

DEFAULT_SAMPLE_SIZE = 5_000
_FINISHED_RESULTS = {"1-0", "0-1", "1/2-1/2"}


def _parse_int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def collect_header_stats(pgn_path: Path) -> dict:
    """Single fast pass over headers only -- no chess move parsing."""
    game_count = 0
    result_counts: Counter[str] = Counter()
    time_control_valid_count = 0
    elos: list[int] = []
    dates: list[str] = []

    with pgn_path.open("r", encoding="utf-8") as stream:
        while True:
            block = read_next_game_block(stream)
            if block is None:
                break
            headers, _lines = block
            game_count += 1

            result_counts[headers.get("Result", "?")] += 1

            white_elo = _parse_int(headers.get("WhiteElo"))
            black_elo = _parse_int(headers.get("BlackElo"))
            if white_elo is not None:
                elos.append(white_elo)
            if black_elo is not None:
                elos.append(black_elo)

            date = headers.get("UTCDate")
            if date:
                dates.append(date)

            if headers.get("TimeControl"):
                time_control_valid_count += 1

    return {
        "game_count": game_count,
        "result_counts": result_counts,
        "elos": elos,
        "dates": dates,
        "time_control_valid_count": time_control_valid_count,
    }


def estimate_avg_plies(
    pgn_path: Path, sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = 0
) -> tuple[float, int]:
    """Reservoir-sample sample_size finished games and replay them for real.

    Returns (average_plies_per_finished_game, games_actually_sampled).
    """
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0

    with pgn_path.open("r", encoding="utf-8") as stream:
        while True:
            block = read_next_game_block(stream)
            if block is None:
                break
            headers, lines = block
            if headers.get("Result") not in _FINISHED_RESULTS:
                continue  # only sample from games that would actually be used

            seen += 1
            text = "".join(lines)
            if len(reservoir) < sample_size:
                reservoir.append(text)
            else:
                j = rng.randrange(seen)
                if j < sample_size:
                    reservoir[j] = text

    ply_counts = []
    for text in reservoir:
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            continue
        ply_counts.append(sum(1 for _ in game.mainline_moves()))

    avg = sum(ply_counts) / len(ply_counts) if ply_counts else 0.0
    return avg, len(ply_counts)


def render_card(
    pgn_path: Path,
    dvc_md5: str,
    source_month: str,
    min_elo: int,
    min_time_control_seconds: int,
    generated_date: str,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> str:
    stats = collect_header_stats(pgn_path)
    avg_plies, sampled = estimate_avg_plies(pgn_path, sample_size=sample_size)

    game_count = stats["game_count"]
    results = stats["result_counts"]
    finished = results.get("1-0", 0) + results.get("0-1", 0) + results.get("1/2-1/2", 0)
    unfinished = game_count - finished
    estimated_positions = round(avg_plies * finished)

    elos = stats["elos"]
    dates = sorted(stats["dates"])
    size_bytes = pgn_path.stat().st_size

    def pct(n: int) -> str:
        return f"{100 * n / game_count:.1f}%" if game_count else "0%"

    lines = [
        f"# Dataset card: {pgn_path.stem}",
        "",
        f"**Source file:** `{pgn_path.as_posix()}` (DVC-tracked, `md5:{dvc_md5}`)",
        f"**Generated:** {generated_date}",
        "**Pipeline:** `dvc.yaml`'s `ingest` stage — see "
        "[`docs/DEVELOPMENT_PIPELINE.md`](../DEVELOPMENT_PIPELINE.md) Phase 01",
        "",
        "## Filters applied",
        "",
        f"- Source: Lichess standard-rated monthly dump, {source_month} "
        f"(`lichess_db_standard_rated_{source_month}.pgn.zst`)",
        f"- Both players rated ≥{min_elo} Elo",
        f"- Time control ≥{min_time_control_seconds} seconds (bullet excluded)",
        "- Standard variant only (guaranteed by the source dump, not a filter step)",
        "",
        "## Summary statistics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| File size | {size_bytes / 1_000_000:.1f} MB ({size_bytes:,} bytes) |",
        f"| Games | {game_count:,} |",
        f"| Date range | {dates[0] if dates else '?'} to {dates[-1] if dates else '?'} |",
        f"| Elo range (both players pooled) | {min(elos)}"
        f"–{max(elos)}, mean {sum(elos) / len(elos):.0f} |"
        if elos
        else "| Elo range | no data |",
        f"| Result: White wins | {results.get('1-0', 0):,} ({pct(results.get('1-0', 0))}) |",
        f"| Result: Black wins | {results.get('0-1', 0):,} ({pct(results.get('0-1', 0))}) |",
        f"| Result: Draws | {results.get('1/2-1/2', 0):,} ({pct(results.get('1/2-1/2', 0))}) |",
        f"| Result: Unfinished (`*`) | {unfinished:,} ({pct(unfinished)}) |",
        f"| Estimated usable positions | ~{estimated_positions:,} "
        f"(finished games only, avg {avg_plies:.1f} plies/game) |",
        "",
        "## Known caveats",
        "",
        "- Ingestion filters on Elo and time control only, not game completion — "
        f"the {unfinished:,} unfinished games above are still in the file, and "
        "contribute 0 training positions when consumed via `ChessPositionDataset`, "
        "which skips them.",
        f"- The position count is an estimate, not exact: {sampled:,} finished "
        "games were fully replayed via python-chess to measure average game "
        f"length, then extrapolated to all {finished:,} finished games. An exact "
        "count would require replaying all of them, which is the same "
        "per-game cost that made the original ingestion slow (~15 minutes for "
        "this many games) — not worth paying just for documentation.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn_path", type=Path)
    parser.add_argument("--dvc-md5", required=True)
    parser.add_argument("--source-month", required=True, help="YYYY-MM")
    parser.add_argument("--min-elo", type=int, required=True)
    parser.add_argument("--min-time-control", type=int, required=True)
    parser.add_argument("--generated-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    card = render_card(
        args.pgn_path,
        dvc_md5=args.dvc_md5,
        source_month=args.source_month,
        min_elo=args.min_elo,
        min_time_control_seconds=args.min_time_control,
        generated_date=args.generated_date,
        sample_size=args.sample_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(card, encoding="utf-8")
    print(f"Dataset card written to {args.output}")


if __name__ == "__main__":
    main()
