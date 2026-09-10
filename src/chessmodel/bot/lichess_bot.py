"""Lichess bot client: plays real games by calling the Phase 04 serving API.

A separate process from the API itself -- this only ever makes HTTP calls
(to Lichess and to our own /move endpoint), never touches the model
directly, so it needs neither torch nor fastapi.

Game-over detection is Lichess's call, not ours: gameState.status is
authoritative for checkmate, stalemate, threefold repetition, the 50-move
rule, etc. -- we just trust it rather than re-deriving it locally.

Concurrency: the event stream (accept/decline challenges, notice new games)
runs on the main thread; each active game gets its own thread consuming its
own per-game stream, so one long game doesn't block new challenges from
being noticed and accepted. httpx.Client and redis.Redis are both documented
thread-safe, so a single shared instance of each is used across threads.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import chess
import httpx
import redis

FINISHED_GAMES_QUEUE_KEY = "chessmodel:finished_games"


class LichessClient:
    def __init__(self, token: str, base_url: str = "https://lichess.org") -> None:
        headers = {"Authorization": f"Bearer {token}"}
        # Two separate clients, not one shared between everything: the event
        # and per-game streams hold a connection open for the entire game,
        # and sharing that client with short request/response calls (accept,
        # decline, move) let the always-open stream contend with -- and
        # measurably delay, several seconds observed live -- time-sensitive
        # one-off calls like accepting a challenge before it expires.
        self._stream_client = httpx.Client(base_url=base_url, headers=headers, timeout=None)
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=10.0)

    def get_account_username(self) -> str:
        response = self._client.get("/api/account")
        response.raise_for_status()
        return str(response.json()["username"])

    def stream_events(self) -> Iterator[dict[str, Any]]:
        with self._stream_client.stream("GET", "/api/stream/event") as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def stream_game(self, game_id: str) -> Iterator[dict[str, Any]]:
        with self._stream_client.stream("GET", f"/api/bot/game/stream/{game_id}") as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.strip():
                    yield json.loads(line)

    def accept_challenge(self, challenge_id: str) -> None:
        response = self._client.post(f"/api/challenge/{challenge_id}/accept")
        response.raise_for_status()

    def decline_challenge(self, challenge_id: str, reason: str = "generic") -> None:
        response = self._client.post(
            f"/api/challenge/{challenge_id}/decline", data={"reason": reason}
        )
        response.raise_for_status()

    def make_move(self, game_id: str, move_uci: str) -> None:
        response = self._client.post(f"/api/bot/game/{game_id}/move/{move_uci}")
        response.raise_for_status()

    def export_pgn(self, game_id: str) -> str:
        response = self._client.get(f"/game/export/{game_id}")
        response.raise_for_status()
        return response.text


@dataclass
class BotClients:
    lichess: LichessClient
    api: httpx.Client  # our own /move endpoint
    redis: redis.Redis
    our_username: str


def should_accept(challenge: dict[str, Any]) -> bool:
    """Only standard chess, per the Phase 05 spec -- no Chess960/Atomic/etc."""
    return bool(challenge.get("variant", {}).get("key") == "standard")


def reconstruct_board(initial_fen: str | None, moves: str) -> chess.Board:
    """Replay a Lichess gameState's space-separated UCI move list onto a
    board, so we have a real FEN to hand to our own /move endpoint."""
    board = (
        chess.Board() if not initial_fen or initial_fen == "startpos" else chess.Board(initial_fen)
    )
    for uci in moves.split():
        board.push(chess.Move.from_uci(uci))
    return board


def _select_move(clients: BotClients, fen: str) -> str:
    response = clients.api.post("/move", json={"fen": fen})
    response.raise_for_status()
    return str(response.json()["move_uci"])


def _push_finished_game(clients: BotClients, game_id: str) -> None:
    pgn = clients.lichess.export_pgn(game_id)
    payload = json.dumps({"game_id": game_id, "pgn": pgn})
    clients.redis.lpush(FINISHED_GAMES_QUEUE_KEY, payload)


def advance_game(
    clients: BotClients,
    game_id: str,
    our_color: chess.Color,
    initial_fen: str | None,
    state: dict[str, Any],
) -> bool:
    """Handle one gameState event. Returns False once the game has ended.

    "Ended" is whatever Lichess's own status field says -- see the module
    docstring on why we don't try to re-derive game-over conditions here.
    """
    if state["status"] != "started":
        _push_finished_game(clients, game_id)
        return False

    board = reconstruct_board(initial_fen, state["moves"])
    if board.turn != our_color:
        return True

    fen = board.fen()
    start = time.monotonic()
    move_uci = _select_move(clients, fen)
    elapsed_ms = (time.monotonic() - start) * 1000

    # No adaptive search-depth budgeting here -- a single forward pass is
    # already ~tens of ms (see the Phase 04 latency benchmarks), so this is
    # a safety-floor warning, not a real scheduling algorithm.
    remaining_ms = state["wtime"] if our_color == chess.WHITE else state["btime"]
    if elapsed_ms > remaining_ms * 0.5:
        print(
            f"WARNING: move took {elapsed_ms:.0f}ms against {remaining_ms}ms remaining on the clock"
        )

    print(f"Game {game_id}: submitting {move_uci} for fen={fen}")
    clients.lichess.make_move(game_id, move_uci)
    return True


def play_game(clients: BotClients, game_id: str) -> None:
    stream = clients.lichess.stream_game(game_id)
    game_full = next(stream)

    # Lichess's "id" fields are always lowercase (the canonical account id),
    # while our_username preserves the registered display casing -- compare
    # case-insensitively or this silently always resolves to BLACK.
    white_id = (game_full["white"].get("id") or "").lower()
    our_color = chess.WHITE if white_id == clients.our_username.lower() else chess.BLACK
    initial_fen = game_full.get("initialFen")
    print(
        f"Game {game_id}: white={game_full['white'].get('id')} "
        f"black={game_full['black'].get('id')} we_are={'white' if our_color else 'black'} "
        f"initial_state={game_full['state']}"
    )

    if not advance_game(clients, game_id, our_color, initial_fen, game_full["state"]):
        return

    for event in stream:
        if event.get("type") != "gameState":
            continue
        print(f"Game {game_id}: gameState={event}")
        if not advance_game(clients, game_id, our_color, initial_fen, event):
            break


def _run_game(clients: BotClients, game_id: str) -> None:
    try:
        play_game(clients, game_id)
    except Exception as exc:
        # A crash in one game must not take down the event loop, which is
        # still tracking/accepting challenges on the main thread.
        print(f"Game {game_id} ended with an error: {exc}")


def _handle_challenge(clients: BotClients, challenge: dict[str, Any]) -> None:
    challenge_id = challenge["id"]
    received_at = time.monotonic()
    print(
        f"Challenge {challenge_id} received: variant={challenge.get('variant', {}).get('key')} "
        f"speed={challenge.get('speed')} rated={challenge.get('rated')} "
        f"timeControl={challenge.get('timeControl')}"
    )
    try:
        if should_accept(challenge):
            clients.lichess.accept_challenge(challenge_id)
            elapsed_ms = (time.monotonic() - received_at) * 1000
            print(f"Challenge {challenge_id} accepted ({elapsed_ms:.0f}ms after receipt)")
        else:
            clients.lichess.decline_challenge(challenge_id, reason="standard")
            print(f"Challenge {challenge_id} declined (non-standard variant)")
    except Exception as exc:
        elapsed_ms = (time.monotonic() - received_at) * 1000
        # A challenge can expire/get withdrawn between appearing on the
        # event stream and us responding to it (e.g. the bot was briefly
        # down) -- one stale challenge must not take down the whole event
        # loop, same reasoning as _run_game's per-game try/except.
        print(
            f"Failed to respond to challenge {challenge_id} "
            f"({elapsed_ms:.0f}ms after receipt): {exc}"
        )


def handle_events(clients: BotClients) -> None:
    for event in clients.lichess.stream_events():
        event_type = event.get("type")
        if event_type == "challenge":
            _handle_challenge(clients, event["challenge"])
        elif event_type == "gameStart":
            game_id = event["game"]["id"]
            print(f"gameStart received for {game_id}")
            threading.Thread(target=_run_game, args=(clients, game_id), daemon=True).start()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-base-url", default=os.environ.get("CHESSMODEL_API_URL", "http://localhost:8000")
    )
    parser.add_argument(
        "--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )
    args = parser.parse_args()

    token = os.environ.get("LICHESS_TOKEN")
    if not token:
        raise RuntimeError("LICHESS_TOKEN environment variable must be set")

    lichess_client = LichessClient(token)
    our_username = lichess_client.get_account_username()
    print(f"Logged in as {our_username}")

    clients = BotClients(
        lichess=lichess_client,
        api=httpx.Client(base_url=args.api_base_url, timeout=10.0),
        redis=redis.from_url(args.redis_url),
        our_username=our_username,
    )
    handle_events(clients)


if __name__ == "__main__":
    main()
