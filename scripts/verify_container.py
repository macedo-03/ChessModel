"""CI container-integration check: poll for health, hit /move, verify schema
and the latency budget against an already-running container.

Deliberately stdlib-only (urllib, not requests/httpx) -- this runs against a
real running server, unlike tests/test_api.py's in-process TestClient checks,
so it doesn't need the project's own dependencies installed to execute.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request

LATENCY_BUDGET_SECONDS = 0.2


def wait_for_health(base_url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError(f"{base_url}/health did not become healthy within {timeout_seconds}s")


def verify_move(base_url: str) -> None:
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    payload = json.dumps({"fen": starting_fen}).encode()
    request = urllib.request.Request(
        f"{base_url}/move", data=payload, headers={"Content-Type": "application/json"}
    )

    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=5) as response:
        elapsed = time.monotonic() - start
        body = json.loads(response.read())
        status = response.status

    assert status == 200, f"Expected 200, got {status}: {body}"
    assert "move_uci" in body, f"Missing move_uci in response: {body}"
    assert "move_san" in body, f"Missing move_san in response: {body}"
    assert "eval" in body, f"Missing eval in response: {body}"
    assert -1.0 <= body["eval"] <= 1.0, f"eval out of range: {body['eval']}"
    assert elapsed < LATENCY_BUDGET_SECONDS, (
        f"/move took {elapsed * 1000:.1f}ms, over the {LATENCY_BUDGET_SECONDS * 1000:.0f}ms budget"
    )
    print(f"/move OK: {body['move_uci']} in {elapsed * 1000:.1f}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    wait_for_health(args.base_url)
    print("Health check OK")
    verify_move(args.base_url)


if __name__ == "__main__":
    main()
