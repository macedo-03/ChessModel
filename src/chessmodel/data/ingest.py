"""Stream-filter a Lichess monthly PGN dump by rating and time control."""

from __future__ import annotations

import argparse
import io
import re
from collections.abc import Mapping
from pathlib import Path
from typing import TextIO

import requests
import zstandard
from tqdm import tqdm

LICHESS_DUMP_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_{month}.pgn.zst"

DEFAULT_MIN_ELO = 2200
DEFAULT_MIN_TIME_CONTROL_SECONDS = 180
DEFAULT_MAX_GAMES = 200_000

_HEADER_LINE_RE = re.compile(r'^\[(\S+)\s+"(.*)"\]\s*$')


def build_dump_url(month: str) -> str:
    """month: 'YYYY-MM', e.g. '2026-07'."""
    return LICHESS_DUMP_URL.format(month=month)


def is_valid_time_control(
    time_control: str, min_seconds: int = DEFAULT_MIN_TIME_CONTROL_SECONDS
) -> bool:
    """TimeControl header is '<base_seconds>+<increment>', or '-' for correspondence/unlimited."""
    try:
        base_seconds = int(time_control.split("+")[0])
    except (ValueError, AttributeError):
        return False
    return base_seconds >= min_seconds


def game_qualifies(
    headers: Mapping[str, str],
    min_elo: int = DEFAULT_MIN_ELO,
    min_time_control_seconds: int = DEFAULT_MIN_TIME_CONTROL_SECONDS,
) -> bool:
    """Both players must meet the Elo bar; bullet-speed games are excluded."""
    try:
        white_elo = int(headers.get("WhiteElo", ""))
        black_elo = int(headers.get("BlackElo", ""))
    except ValueError:
        return False

    if white_elo < min_elo or black_elo < min_elo:
        return False

    return is_valid_time_control(headers.get("TimeControl", ""), min_time_control_seconds)


def _read_next_game_block(stream: TextIO) -> tuple[dict[str, str], list[str]] | None:
    """Read one game's headers and raw text (headers + movetext) from stream.

    Pure text scanning — no chess logic, no move parsing. Returns None at EOF.
    The raw lines are exactly what should be written out verbatim for a
    qualifying game, byte-for-byte unchanged from the source.
    """
    headers: dict[str, str] = {}
    lines: list[str] = []

    line = stream.readline()
    while line and line.strip() == "":  # skip blank lines between games
        line = stream.readline()
    if not line:
        return None  # EOF, no more games

    while line and line.strip() != "":  # header block
        lines.append(line)
        match = _HEADER_LINE_RE.match(line.strip())
        if match:
            headers[match.group(1)] = match.group(2)
        line = stream.readline()

    if line:  # blank line separating headers from movetext
        lines.append(line)
        line = stream.readline()

    while line and line.strip() != "":  # movetext block
        lines.append(line)
        line = stream.readline()

    if line:  # trailing blank line before the next game
        lines.append(line)

    return headers, lines


def filter_games(
    pgn_stream: TextIO,
    output_stream: TextIO,
    min_elo: int = DEFAULT_MIN_ELO,
    min_time_control_seconds: int = DEFAULT_MIN_TIME_CONTROL_SECONDS,
    max_games: int | None = DEFAULT_MAX_GAMES,
    max_scanned: int | None = None,
) -> tuple[int, int]:
    """Read games one at a time from pgn_stream, write qualifying ones to output_stream.

    max_games caps how many *qualifying* games to keep. max_scanned independently
    caps how many games to look at regardless of how many qualified — a strict
    rating filter can have a very low hit rate, so max_games alone doesn't bound
    how much of the source stream gets consumed. Use max_scanned for a bounded
    smoke test against the real dump.

    Returns (scanned_count, kept_count).
    """
    scanned = 0
    kept = 0

    with tqdm(desc="Scanning games", unit="games") as progress:
        while (max_games is None or kept < max_games) and (
            max_scanned is None or scanned < max_scanned
        ):
            block = _read_next_game_block(pgn_stream)
            if block is None:
                break

            headers, lines = block
            scanned += 1
            progress.update(1)

            if game_qualifies(headers, min_elo, min_time_control_seconds):
                output_stream.writelines(lines)
                kept += 1
                progress.set_postfix(kept=kept)

    return scanned, kept


def download_and_filter(
    month: str,
    output_path: Path,
    min_elo: int = DEFAULT_MIN_ELO,
    min_time_control_seconds: int = DEFAULT_MIN_TIME_CONTROL_SECONDS,
    max_games: int | None = DEFAULT_MAX_GAMES,
    max_scanned: int | None = None,
) -> tuple[int, int]:
    """Stream the monthly dump from Lichess straight through to a filtered PGN file."""
    url = build_dump_url(month)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(response.raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            with output_path.open("w", encoding="utf-8") as out:
                return filter_games(
                    text_stream, out, min_elo, min_time_control_seconds, max_games, max_scanned
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, help="YYYY-MM, e.g. 2026-07")
    parser.add_argument("--min-elo", type=int, default=DEFAULT_MIN_ELO)
    parser.add_argument(
        "--min-time-control",
        type=int,
        default=DEFAULT_MIN_TIME_CONTROL_SECONDS,
        help="seconds",
    )
    parser.add_argument("--max-games", type=int, default=DEFAULT_MAX_GAMES)
    parser.add_argument(
        "--max-scanned",
        type=int,
        default=None,
        help="Stop after looking at this many games, regardless of how many qualified. "
        "Useful for a bounded smoke test against the real dump.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to data/raw/lichess_<month>_elo<min-elo>.pgn",
    )
    args = parser.parse_args()

    output_path = args.output or Path(f"data/raw/lichess_{args.month}_elo{args.min_elo}.pgn")

    scanned, kept = download_and_filter(
        args.month,
        output_path,
        args.min_elo,
        args.min_time_control,
        args.max_games,
        args.max_scanned,
    )
    print(f"Scanned {scanned} games, kept {kept} qualifying games -> {output_path}")


if __name__ == "__main__":
    main()
